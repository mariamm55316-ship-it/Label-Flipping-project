import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# tell Python where preprocessing.py is (in the src folder), then import it
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocessing as pp  # noqa: E402

# the 8 categories we expect, in alphabetical order (used in many tests)
CATEGORIES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]

# ---------------------------------------------------------------------------
# 1. Mapping: the 34 labels must go into the right 8 categories
# Why: if a label goes to the wrong category, every count after it is wrong.
# ---------------------------------------------------------------------------
def test_mapping():
    assert len(pp.LABEL_TO_CATEGORY) == 34                       # all 34 labels exist
    assert sorted(set(pp.LABEL_TO_CATEGORY.values())) == CATEGORIES   # exactly 8 categories
    assert pp.map_to_category("DDoS-UDP_Flood") == "DDoS"        # a normal label
    assert pp.map_to_category(" BenignTraffic ") == "Benign"     # extra spaces are ignored


def test_unknown_label_gives_error():
    # a label that is not in the dictionary must stop the program,
    # not be silently skipped (that would lose rows without us knowing)
    with pytest.raises(ValueError):
        pp.map_to_category("Typo-Attack")


# ---------------------------------------------------------------------------
# 2. Keep chance: target / total * margin, and never above 1
# Why: this decides how many rows we keep from each category.
# ---------------------------------------------------------------------------
def test_keep_probs():
    counts = {"DDoS": 1000, "Web": 10}        # pretend raw row counts
    targets = {"DDoS": 100, "Web": 100}       # how many rows we want
    probs = pp.make_keep_probs(counts, targets, margin=1.5)

    assert probs["DDoS"] == pytest.approx(0.15)   # 100 / 1000 * 1.5 = 0.15
    assert probs["Web"] == 1.0                    # 100 / 10 * 1.5 = 15, a chance can't be
                                                  # above 1, so it is capped (keep every row)


# ---------------------------------------------------------------------------
# 3. Cleaning: remove inf, NaN and duplicate rows
# Why: models crash on inf/NaN, and duplicates can end up in both train and test.
# ---------------------------------------------------------------------------
def test_clean():
    # 6 rows: 1.0 ok, 2.0 ok, 2.0 copy, inf bad, NaN bad, 5.0 ok  -> 3 should be left
    df = pd.DataFrame({"f0": [1.0, 2.0, 2.0, np.inf, np.nan, 5.0],
                       "category": ["Benign"] * 6})
    out, info = pp.clean(df)

    assert len(out) == 3                       # 3 good rows left
    assert info["removed_inf_or_nan"] == 2     # the inf row and the NaN row
    assert info["removed_duplicates"] == 1     # the copy of 2.0


# ---------------------------------------------------------------------------
# 4. Trimming: each category is cut to exactly its target size
# Why: the report says 13,000 per attack category; this proves the cut is exact.
# ---------------------------------------------------------------------------
def test_trim():
    df = pd.DataFrame({"f0": np.arange(50.0), "category": "DDoS"})   # 50 DDoS rows
    out, shortfalls = pp.trim_to_target(df, {"DDoS": 20})            # we want 20

    assert len(out) == 20            # exactly 20 kept
    assert shortfalls == {}          # no category was too small


# ---------------------------------------------------------------------------
# 5. Labels: Benign = 0, Attack = 1, and the answer is never a feature
# Why: if "label" or "category" stayed in X, the model would see the answer
# and get a fake perfect score.
# ---------------------------------------------------------------------------
def test_labels():
    df = pd.DataFrame({"f0": [1.0, 2.0],
                       "label": ["BenignTraffic", "XSS"],
                       "category": ["Benign", "Web"]})
    X, y, cat = pp.make_labels(df)

    assert list(X.columns) == ["f0"]   # only the real feature, no label/category
    assert list(y) == [0, 1]           # Benign -> 0, XSS (an attack) -> 1


# ---------------------------------------------------------------------------
# 6. Split and scale
# A fixture is shared setup: these 3 tests all use the same fake data.
# 8 categories x 50 rows = 400 rows, values around 10 (so scaling really changes them)
# ---------------------------------------------------------------------------
@pytest.fixture
def split():
    cat = pd.Series(np.repeat(CATEGORIES, 50))
    X = pd.DataFrame(np.random.default_rng(1).normal(10, 3, size=(400, 3)))
    y = (cat != "Benign").astype(int)
    return pp.split_and_scale(X, y, cat)     # gives back (arrays, scaler)


def test_split_80_20_and_stratified(split):
    arrays, _ = split
    # 80% of 400 = 320 train, 20% = 80 test
    assert len(arrays["X_train"]) == 320 and len(arrays["X_test"]) == 80
    # stratified: each category keeps its share, so 50 x 20% = 10 per category in test
    names, counts = np.unique(arrays["cat_test"], return_counts=True)
    assert list(counts) == [10] * 8


def test_scaler_learns_from_train_only(split):
    _, scaler = split
    # the scaler must only see the 320 train rows, not all 400.
    # Why: if it saw the test rows, test information would leak into training.
    assert scaler.n_samples_seen_ == 320


def test_types(split):
    arrays, _ = split
    assert arrays["X_train"].dtype == np.float64   # float64, see the next test for why
    assert arrays["y_train"].dtype == np.int8      # y only holds 0/1, so int8 is enough


# ---------------------------------------------------------------------------
# 7. The float32 bug: rows that differ only a tiny bit must stay different
# Why: with float32, big values + 0.01 were rounded to the same number,
# which made fake duplicates and train/test overlap. This test locks the fix.
# ---------------------------------------------------------------------------
def test_near_identical_rows_stay_different():
    # 200 big values (around 83 million), each written twice -> 400 rows
    X = pd.DataFrame({"f0": np.repeat(8.3e7 + np.arange(200) * 50_000.0, 2)})
    X.loc[1::2, "f0"] += 0.01          # add 0.01 to every second row -> 400 different rows
    cat = pd.Series(np.tile(CATEGORIES, 50))
    y = (cat != "Benign").astype(int)

    arrays, _ = pp.split_and_scale(X, y, cat)
    all_rows = np.vstack([arrays["X_train"], arrays["X_test"]])   # put train and test together
    assert len(np.unique(all_rows, axis=0)) == 400   # still 400 different rows (float32 gave fewer)


# ---------------------------------------------------------------------------
# 8. The whole pipeline, start to end, on small fake csv files
# Why: each step can work alone but still break when joined together.
# ---------------------------------------------------------------------------
def make_fake_csvs(folder):
    """Write 2 csv files. Each has all 34 labels, 60 random rows per label."""
    rng = np.random.default_rng(0)
    for k in range(2):
        rows = []
        for label in pp.LABEL_TO_CATEGORY:
            part = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "d"])
            part["label"] = label
            rows.append(part)
        pd.concat(rows).to_csv(folder / f"part{k}.csv", index=False)


def test_pipeline(tmp_path):
    # tmp_path = an empty temporary folder from pytest, deleted after the test
    make_fake_csvs(tmp_path)
    # small targets so it runs in a second: 30 per attack category, 80 Benign
    arrays = pp.run(tmp_path, tmp_path / "out", target_attack=30, target_benign=80, margin=3.0)

    # 7 x 30 + 80 = 290 rows -> 80% = 232 train, 20% = 58 test
    assert len(arrays["X_train"]) == 232 and len(arrays["X_test"]) == 58
    assert (tmp_path / "out" / "summary.json").exists()   # the output files were saved

    # no row may be in both train and test (tobytes turns a row into something a set can hold)
    train = {row.tobytes() for row in arrays["X_train"]}
    test = {row.tobytes() for row in arrays["X_test"]}
    assert train.isdisjoint(test)

    # run everything again: seed 42 must give exactly the same result
    again = pp.run(tmp_path, tmp_path / "out2", target_attack=30, target_benign=80, margin=3.0)
    assert np.array_equal(arrays["X_train"], again["X_train"])


def test_empty_folder_gives_error(tmp_path):
    # a folder with no csv files must give a clear error, not empty outputs
    with pytest.raises(FileNotFoundError):
        pp.run(tmp_path, tmp_path / "out")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])