"""Run label-poisoning experiments and save results."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from baselines import make_models
from defence import defend
from metrics import classification_metrics, defence_metrics, attack_success_rate
from poisoning import poison_labels

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_RATES = [.05, .10, .20]
DEFAULT_COUNTS = [520, 1040, 2080]
MODELS = ["RF", "LR"]
KEY_COLUMNS = ["seed", "comparison", "strategy", "n_flipped", "model", "condition"]


def score_predictions(y_true, y_pred, cat_test, strategy="none"):
    scores = classification_metrics(y_true, y_pred, cat_test)
    scores["ddos_miss_rate"] = attack_success_rate(y_true, y_pred, cat_test)
    scores["asr_DDoS"] = scores["ddos_miss_rate"] if strategy == "targeted" else np.nan
    return scores


def atomic_csv(rows, path):
    """Write checkpoints atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def row_key(row, defence=False):
    key = (int(row["seed"]), row["comparison"], row["strategy"], int(row["n_flipped"]))
    return key if defence else key + (row["model"], row["condition"])


def cases(config):
    standard = [("standard", s, r, None) for s in ["random", "targeted"] for r in config["rates"]]
    return standard + [("equal_flips", "random", None, n) for n in config["equal_flips"]]


def expected_keys(config, counts):
    model_keys, defence_keys, rates = set(), set(), {}
    for seed in config["seeds"]:
        for name in MODELS:
            model_keys.add((seed, "clean", "none", 0, name, "clean"))
        for comparison, strategy, rate, count in cases(config):
            pool = counts["n_attack"] if strategy == "random" else counts["n_target"]
            flips = int(count) if count is not None else int(round(rate * pool))
            if flips > pool:
                raise ValueError(f"Cannot select {flips} labels from a pool of {pool}")
            dk = (seed, comparison, strategy, flips)
            if dk in defence_keys:
                raise ValueError("Requested settings produce duplicate experiment keys")
            defence_keys.add(dk)
            rates[dk] = rate
            for name in MODELS:
                for condition in ["poisoned", "defended"]:
                    model_keys.add(dk + (name, condition))
    return model_keys, defence_keys, rates


def validate_tables(models, defences, config, counts, complete=True):
    """Validate result coverage, metrics and sample counts."""
    expected_m, expected_d, case_rates = expected_keys(config, counts)
    seen_m, seen_d, defence_lookup = set(), set(), {}
    n = counts["n_train"]
    for row in defences:
        k = row_key(row, defence=True)
        if k in seen_d or k not in expected_d:
            raise ValueError(f"Duplicate or unexpected defence result: {k}")
        seen_d.add(k)
        defence_lookup[k] = row
        f, m, c = [int(row[x]) for x in ["n_flipped", "n_removed", "n_caught"]]
        if int(row["n_train"]) != n or not (0 <= c <= min(f, m) <= n) or m - c > n - f:
            raise ValueError(f"Invalid defence counts: {k}")
        expected = {
            "poison_detection_rate": c / f if f else np.nan,
            "removal_precision": c / m if m else np.nan,
            "clean_removal_rate": (m - c) / (n - f) if n - f else np.nan,
        }
        for metric, value in expected.items():
            if not np.isclose(float(row[metric]), value, equal_nan=True):
                raise ValueError(f"Invalid defence metric {metric}: {k}")
    for row in models:
        k = row_key(row)
        if k in seen_m or k not in expected_m:
            raise ValueError(f"Duplicate or unexpected model result: {k}")
        seen_m.add(k)
        for metric in ["accuracy", "precision", "recall", "f1", "fpr"] + ["recall_" + c for c in counts["attack_categories"]]:
            value = float(row[metric])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid model metric {metric}: {k}")
        target_miss = 1.0 - float(row["recall_DDoS"])
        if not np.isclose(float(row["ddos_miss_rate"]), target_miss):
            raise ValueError(f"Incorrect DDoS miss rate: {k}")
        asr = float(row["asr_DDoS"])
        if row["strategy"] == "targeted":
            if not np.isclose(asr, target_miss):
                raise ValueError(f"Incorrect targeted ASR: {k}")
        elif not np.isnan(asr):
            raise ValueError(f"Targeted ASR should be empty for non-targeted runs: {k}")
        rate = 0. if row["condition"] == "clean" else case_rates[k[:4]]
        saved_rate = row.get("rate")
        if rate is None:
            if not pd.isna(saved_rate):
                raise ValueError(f"Equal-flip rate should be empty: {k}")
        elif not np.isclose(float(saved_rate), rate):
            raise ValueError(f"Incorrect poisoning rate: {k}")
        if row["condition"] == "defended":
            if k[:4] not in defence_lookup:
                raise ValueError(f"Defended model has no matching defence record: {k}")
            used = n - int(defence_lookup[k[:4]]["n_removed"])
        else:
            used = n
        if int(row["n_training_used"]) != used:
            raise ValueError(f"Incorrect training size: {k}")
    if complete and (seen_m != expected_m or seen_d != expected_d):
        raise ValueError(f"Incomplete grid: missing {len(expected_m-seen_m)} models and {len(expected_d-seen_d)} defences")
    return {"model_rows": len(seen_m), "defence_rows": len(seen_d), "complete": seen_m == expected_m and seen_d == expected_d}


def validate_data(data):
    X, Xt, y, yt, cat, catt, neighbours = [data[k] for k in ["X_train", "X_test", "y_train", "y_test", "cat_train", "cat_test", "neighbours"]]
    if X.ndim != 2 or Xt.ndim != 2 or X.shape[1] != Xt.shape[1] or not len(X) or not len(Xt):
        raise ValueError("Training and test features must be non-empty matrices with matching columns")
    for features, labels, categories in [(X, y, cat), (Xt, yt, catt)]:
        if labels.ndim != 1 or categories.ndim != 1 or not len(features) == len(labels) == len(categories):
            raise ValueError("Features, labels, and categories must have matching row counts")
        if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("Features must be finite and labels binary")
        if not np.array_equal(labels == 0, categories == "Benign"):
            raise ValueError("Categories and labels do not agree")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Both classes must exist in training and test data")
    if neighbours.shape != (len(y), 10) or neighbours.dtype.kind not in "iu":
        raise ValueError("The cached neighbours must have shape (n_train, 10) and integer entries")
    if np.any(neighbours < 0) or np.any(neighbours >= len(y)) or np.any(neighbours == np.arange(len(y))[:, None]):
        raise ValueError("Cached neighbours contain invalid indices or self-matches")
    if np.any(np.diff(np.sort(neighbours, axis=1), axis=1) == 0):
        raise ValueError("Cached neighbour rows contain duplicate indices")
    counts = {
        "n_train": len(y), "n_test": len(yt), "n_features": X.shape[1],
        "n_attack": int((y == 1).sum()), "n_target": int((cat == "DDoS").sum()),
        "attack_categories": sorted(set(catt) - {"Benign"}),
    }
    if not counts["n_target"]:
        raise ValueError("DDoS training samples are required")
    return counts


def load_data(path):
    data = {name: np.load(Path(path) / (name + ".npy"), allow_pickle=False) for name in [
        "X_train", "X_test", "y_train", "y_test", "cat_train", "cat_test", "neighbours"]}
    validate_data(data)
    return data


def demo_data():
    """Generate synthetic data for the demo."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    X, y = make_classification(n_samples=1400, n_features=46, n_informative=14,
                              n_redundant=6, weights=[.5, .5], flip_y=0, random_state=42)
    cat = np.full(len(y), "Benign", dtype="U10")
    for category, indices in zip(["BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"], np.array_split(np.flatnonzero(y == 1), 7)):
        cat[indices] = category
    train, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=cat, random_state=42)
    scaler = StandardScaler().fit(X[train])
    X_train, X_test = scaler.transform(X[train]), scaler.transform(X[test])
    # Exclude self-matches.
    neighbours = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(X_train).kneighbors(return_distance=False)
    return dict(X_train=X_train, X_test=X_test, y_train=y[train].astype(np.int8),
                y_test=y[test].astype(np.int8), cat_train=cat[train], cat_test=cat[test], neighbours=neighbours)


def fingerprint(data, config, dataset_kind):
    h = hashlib.sha256(json.dumps({"config": config, "dataset_kind": dataset_kind}, sort_keys=True).encode())
    for name, value in sorted(data.items()):
        h.update(name.encode()); h.update(str(value.shape).encode()); h.update(str(value.dtype).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    for name in ["metrics.py", "baselines.py", "poisoning.py", "defence.py", "run_experiments.py"]:
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def read_records(path):
    return pd.read_csv(path).to_dict("records") if Path(path).exists() else []


def export_compatibility(models, defences, output):
    """Export baseline and defence CSV files."""
    atomic_csv([r for r in models if r["condition"] == "clean"], output / "clean_baselines_all_seeds.csv")
    atomic_csv([r for r in defences if r["comparison"] == "standard"], output / "defence_all_rates_all_seeds.csv")
    atomic_csv([r for r in defences if r["comparison"] == "equal_flips"], output / "defence_equal_flips_all_seeds.csv")


def run(data, output, config, dataset_kind="real", model_factory=make_models):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    counts = validate_data(data)
    expected_keys(config, counts)
    current_fingerprint = fingerprint(data, config, dataset_kind)
    manifest_path = output / "run_manifest.json"
    model_path, defence_path = output / "model_results_all_runs.csv", output / "defence_results.csv"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != current_fingerprint:
            raise ValueError("Data, code, or settings differ from this checkpoint. Use a new --output-dir.")
    else:
        if model_path.exists() or defence_path.exists():
            raise ValueError("Existing results have no matching manifest. Use a new --output-dir.")
        manifest = {"fingerprint": current_fingerprint, "config": config, "counts": counts,
                    "dataset_kind": dataset_kind, "provenance": "Generated by packaged runner",
                    "environment": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn"]}}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    model_rows, defence_rows = read_records(model_path), read_records(defence_path)
    validate_tables(model_rows, defence_rows, config, counts, complete=False)
    completed = {row_key(r) for r in model_rows}
    completed_defences = {row_key(r, True) for r in defence_rows}
    X, y, cat = [data[k] for k in ["X_train", "y_train", "cat_train"]]
    for seed in config["seeds"]:
        for name in MODELS:
            metadata = dict(seed=seed, comparison="clean", strategy="none", rate=0., n_flipped=0, model=name, condition="clean")
            if row_key(metadata) in completed:
                continue
            model = model_factory(seed=seed)[name]
            model.fit(X, y)
            scores = score_predictions(data["y_test"], model.predict(data["X_test"]), data["cat_test"])
            model_rows.append({**metadata, "n_training_used": len(y), **scores})
            completed.add(row_key(metadata)); atomic_csv(model_rows, model_path)
            print(f"Saved clean {name}, seed {seed}", flush=True)
        for comparison, strategy, rate, count in cases(config):
            run_y, flips = poison_labels(y, cat, strategy, rate, seed, n_flips=count)
            metadata = dict(seed=seed, comparison=comparison, strategy=strategy, rate=rate, n_flipped=len(flips))
            dk = row_key(metadata, True)
            pending = [(name, condition) for name in MODELS for condition in ["poisoned", "defended"]
                       if row_key({**metadata, "model": name, "condition": condition}) not in completed]
            if not pending and dk in completed_defences:
                continue
            kept = defend(X, run_y, neighbours=data["neighbours"])
            stats = defence_metrics(len(y), flips, kept)
            if len(np.unique(run_y[kept])) != 2:
                raise ValueError("Defence left fewer than two classes; cannot compare both classifiers")
            if dk not in completed_defences:
                defence_rows.append({**metadata, "n_train": len(y), **stats})
                completed_defences.add(dk); atomic_csv(defence_rows, defence_path)
            for name, condition in pending:
                model = model_factory(seed=seed)[name]
                train_X, train_y = (X, run_y) if condition == "poisoned" else (X[kept], run_y[kept])
                model.fit(train_X, train_y)
                scores = score_predictions(data["y_test"], model.predict(data["X_test"]), data["cat_test"], strategy=strategy)
                row = {**metadata, "model": name, "condition": condition, "n_training_used": len(train_y), **scores}
                model_rows.append(row); completed.add(row_key(row)); atomic_csv(model_rows, model_path)
                print(f"Saved {seed} {comparison} {strategy} {len(flips)} {name} {condition}", flush=True)
    report = validate_tables(model_rows, defence_rows, config, counts)
    export_compatibility(model_rows, defence_rows, output)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--equal-flips", type=int, nargs="+", default=DEFAULT_COUNTS)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--validate-results", action="store_true")
    args = parser.parse_args()
    if args.validate_results:
        manifest = json.loads((args.output_dir / "run_manifest.json").read_text())
        report = validate_tables(read_records(args.output_dir / "model_results_all_runs.csv"),
                                 read_records(args.output_dir / "defence_results.csv"),
                                 manifest["config"], manifest["counts"])
        print(json.dumps(report, indent=2))
        return
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Seeds must be distinct non-negative integers")
    if any(not np.isfinite(r) or not 0 < r <= 1 for r in args.rates):
        parser.error("Rates must be finite fractions in (0, 1]")
    if len(set(args.equal_flips)) != len(args.equal_flips) or any(n <= 0 for n in args.equal_flips):
        parser.error("Equal-flip counts must be distinct positive integers")
    if args.demo and (args.output_dir.resolve() == Path("results").resolve() or args.output_dir.resolve().name == "results"):
        parser.error("Use a separate --output-dir for synthetic demo results")
    config = dict(seeds=args.seeds, rates=args.rates, equal_flips=args.equal_flips)
    data = demo_data() if args.demo else load_data(args.data_dir)
    run(data, args.output_dir, config, "synthetic_demo" if args.demo else "real")


if __name__ == "__main__":
    main()
