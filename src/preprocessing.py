
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Settings (all tunable values live here, in one place)

# Why: a fixed seed makes every random step (sampling, trimming, shuffling,
# splitting) repeatable, so the numbers in the report can be reproduced.
SEED = 42
TARGET_PER_ATTACK = 13_000   # 7 attack categories x 13,000 rows each.

# Why 91,000: it equals 7 x 13,000, so Benign and Attack have the same total
# size. That gives a balanced binary problem (Benign vs Attack).
TARGET_BENIGN = 91_000

MARGIN = 1.5                 # keep 50% extra, because cleaning removes some rows

# Why chunks: the raw files are too big to load at once. Reading 200,000 rows
# at a time keeps memory use small. Lower this value if you run out of RAM.
CHUNK_SIZE = 200_000

LABEL_COL = "label"          # name of the label column in the csv files

# 34 labels -> 8 categories
CATEGORY_LABELS = {
 "DDoS": [
     "DDoS-RSTFINFlood", "DDoS-PSHACK_Flood", "DDoS-SYN_Flood",
     "DDoS-UDP_Flood", "DDoS-TCP_Flood", "DDoS-ICMP_Flood",
     "DDoS-SynonymousIP_Flood", "DDoS-ACK_Fragmentation",
     "DDoS-UDP_Fragmentation", "DDoS-ICMP_Fragmentation",
     "DDoS-SlowLoris", "DDoS-HTTP_Flood",
 ],
 "DoS": ["DoS-UDP_Flood", "DoS-SYN_Flood", "DoS-TCP_Flood", "DoS-HTTP_Flood"],
 "Mirai": ["Mirai-greeth_flood", "Mirai-udpplain", "Mirai-greip_flood"],
 "Recon": [
     "Recon-PortScan", "Recon-OSScan", "Recon-HostDiscovery",
     "Recon-PingSweep", "VulnerabilityScan",
 ],
 "Web": [
     "SqlInjection", "CommandInjection", "Backdoor_Malware",
     "Uploading_Attack", "XSS", "BrowserHijacking",
 ],
 "BruteForce": ["DictionaryBruteForce"],
 "Spoofing": ["MITM-ArpSpoofing", "DNS_Spoofing"],
 "Benign": ["BenignTraffic"],
}

# Invert the dictionary: label -> category (e.g. "DDoS-UDP_Flood" -> "DDoS").
# Why: a flat lookup is a single fast dictionary access for every row.

LABEL_TO_CATEGORY = {}
for cat_name, label_names in CATEGORY_LABELS.items():
 for label_name in label_names:
     LABEL_TO_CATEGORY[label_name] = cat_name


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Small helpers
def map_to_category(label):
 #'DDoS-UDP_Flood' -> 'DDoS'. An unknown label raises an error.
 label = str(label).strip()            # remove accidental spaces
 if label not in LABEL_TO_CATEGORY:

     raise ValueError(f"Unknown label: {label!r}")
 return LABEL_TO_CATEGORY[label]


def read_chunks(path, only_label=False):
 #strip() inside usecols: some csv headers contain stray spaces
 usecols = (lambda name: name.strip() == LABEL_COL) if only_label else None
 for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
     chunk.columns = chunk.columns.str.strip()   # remove spaces in column names
     yield chunk


def get_feature_columns(df):
 # these two columns hold the answer, so they must never be used as features
 result = []                           # start with an empty list
 for c in df.columns:                  # go through every column name
     if c not in (LABEL_COL, "category"):  # skip "label" and "category"
         result.append(c)              # keep all the others
 return result


# ---------------------------------------------------------
# ---------------------------------------------------------
# 1.First step count rows per category

def count_per_category(files):
 #Count how many rows each category has (only the label column is read).

 # 1: count rows per fine-grained label, over all files and chunks
 label_counts = {}            # example: {"XSS": 3846, "BenignTraffic": 98000, ...}
 for path in files:           # go through every csv file
     for chunk in read_chunks(path, only_label=True):   # read it piece by piece, label column only
         for label, n in chunk[LABEL_COL].value_counts().items(): # value_counts() counts how many times each label appears in this chunk
             label_counts[label] = label_counts.get(label, 0) + int(n)

 # 2:add the label counts together into category counts
 category_counts = {}
 for label, n in label_counts.items():
     category = map_to_category(label)     # "DDoS-UDP_Flood" -> "DDoS"
     category_counts[category] = category_counts.get(category, 0) + n
 return category_counts     # total raw rows for each of the 8 categories


#How many rows we want for each category.
def make_targets(target_attack, target_benign):
 """Return the wanted number of rows for each category."""
 targets = {}
 for category in CATEGORY_LABELS:        # go through all 8 categories
     # Benign has its own (larger) target; all attack categories share one.
     targets[category] = target_benign if category == "Benign" else target_attack
 return targets


#Chance of keeping a row = target / total rows of its category * margin.
def make_keep_probs(counts, targets, margin):

 keep_probs = {}
 for category, target in targets.items():
     total = counts.get(category, 0)       # how many raw rows this category has
     if total == 0:
         print(f"WARNING: no rows found for category {category}")
         keep_probs[category] = 0.0
     else:
         keep_probs[category] = min(1.0, target / total * margin)
 return keep_probs



# --------------------------------------------
# --------------------------------------------
# Step 2: keep a random part of every chunk
def sample_chunk(chunk, keep_probs, rng):
 #Add a 'category' column, then keep each row with its category's chance.
 chunk = chunk.copy()           # work on a copy, never change the input
 labels = chunk[LABEL_COL].astype(str).str.strip()     # the label of every row, as clean text
 chunk["category"] = labels.map(LABEL_TO_CATEGORY)

 # a label that is not in the dictionary becomes empty: stop with a clear error
 if chunk["category"].isna().any():
     unknown = sorted(set(labels[chunk["category"].isna()]))   # list the bad labels
     raise ValueError(f"Unknown labels: {unknown}")

 probs = chunk["category"].map(keep_probs).to_numpy()   # each row gets its category's keep chance
 draws = rng.random(len(chunk))        # one random number (0 to 1) per row
 return chunk[draws < probs]           # keep the row if its number is small


def load_sample(files, keep_probs, rng):
 #Go through all the files and join the kept rows into one table.
 pieces = []                             # the kept parts, one per chunk
 for i, path in enumerate(files, 1):
     print(f"[{i}/{len(files)}] {Path(path).name}")       # progress, like [5/169] Merged05.csv
     for chunk in read_chunks(path):                         # read the file piece by piece
         pieces.append(sample_chunk(chunk, keep_probs, rng))       # keep a random part, save it

 if not pieces:                                      # nothing was read at all
     raise ValueError("No rows were read.")
 df = pd.concat(pieces, ignore_index=True)          # glue all the pieces into one table
 print("Rows kept per category:")
 print(df["category"].value_counts().to_string())     # check the counts per category
 return df


# ----------------------------------------
# ----------------------------------------
# Step 3: clean
def clean(df):
 """Replace inf by NaN, drop NaN rows, drop duplicate rows (features only)."""
 feature_cols = get_feature_columns(df)   # the 46 feature columns (no label, no category)
 df = df.copy()                          # work on a copy, never change the input
 start = len(df)                        # how many rows we have before any cleaning

 # 1) inf (infinity) is not a usable number, so turn it into NaN (missing value)
 df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

 # 2) drop every row that has a missing value (NaN) in any feature
 df = df.dropna(subset=feature_cols)
 after_nan = len(df)                   # rows left after removing inf / NaN

 # 3) drop duplicate rows: judged on the features only (not on the label), as in the report
 df = df.drop_duplicates(subset=feature_cols)
 after_dup = len(df)                # rows left after removing duplicates

 # keep the numbers will be helpful for the report
 info = {"rows_before_cleaning": start,
     "removed_inf_or_nan": start - after_nan,
     "removed_duplicates": after_nan - after_dup,
     "rows_after_cleaning": after_dup,
 }
 print("Cleaning:", info)
 # reset_index renumbers the rows 0, 1, 2, ... because removing rows leaves gaps
 return df.reset_index(drop=True), info   # returns TWO things: the clean table and the numbers


# ----------------------------------------
# ----------------------------------------
# Step 4: cut every category to its target

def trim_to_target(df, targets):
 """Too many rows: pick a random subset. Too few: keep all and warn."""
 parts = []
 shortfalls = {}              # categories that ended below their target
 for category, target in targets.items():
     rows = df[df["category"] == category]     # all rows of this one category
     if len(rows) > target:
         # Extra rows come from the margin: pick exactly `target` at random.
         rows = rows.sample(n=target, random_state=SEED)
     elif len(rows) < target:
         shortfalls[category] = {"got": len(rows), "wanted": target}
         print(f"WARNING: {category} has only {len(rows)} rows (target {target}). "
               f"Update the numbers in the report, or use a bigger --margin.")
     parts.append(rows)

 # shuffle once, so the categories are mixed
 df = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
 return df, shortfalls


# ---------------------------------------------------------------------------
# Step 5: labels
# ---------------------------------------------------------------------------
def make_labels(df):
 # Return X (features), y (0 = Benign, 1 = Attack) and cat (category name).
 feature_cols = get_feature_columns(df)    # label and category are NOT features
 X = df[feature_cols]

 cat = df["category"]                     # kept separately for poisoning and per-category recall
 y = (cat != "Benign").astype(int)        # Attack = 1, Benign = 0
 return X, y, cat


# ---------------------------------------------------------------------------
# Step 6: split, then scale
# ---------------------------------------------------------------------------
def split_and_scale(X, y, cat, test_size=0.2):
 #80/20 split (stratified by category). The scaler only sees the train part.
 # split X, y and cat together so the rows stay aligned
 # stratify=cat keeps the same category proportions in train and test
 X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
     X, y, cat, test_size=test_size, stratify=cat, random_state=SEED
 )
 # fit the scaler on train only, so no test information leaks into training

 scaler = StandardScaler()
 X_train = scaler.fit_transform(X_train)
 X_test = scaler.transform(X_test)     # test uses the train mean and std

 # smaller types save memory and disk space
 arrays = {
     "X_train": X_train.astype("float32"),
     "X_test": X_test.astype("float32"),
     "y_train": y_train.to_numpy().astype("int8"),
     "y_test": y_test.to_numpy().astype("int8"),
     "cat_train": cat_train.to_numpy().astype(str),
     "cat_test": cat_test.to_numpy().astype(str),
 }
 return arrays, scaler


# ------------------------------------------
# ------------------------------------------
# Step 7: save
def save_outputs(out_dir, arrays, scaler, feature_names, summary):
 out_dir = Path(out_dir)
 out_dir.mkdir(parents=True, exist_ok=True)   # create the folder if missing
 # save the 6 main arrays as .npy files
 for name, array in arrays.items():
     np.save(out_dir / f"{name}.npy", array)

 # save the scaler so new data can be scaled the same way later
 joblib.dump(scaler, out_dir / "scaler.joblib")
 # save the feature names to keep the column order of X
 (out_dir / "feature_names.json").write_text(json.dumps(list(feature_names), indent=2))

 # save the numbers needed for the report
 (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
 print(f"Saved to {out_dir}")


# ------------------------------------------
# ------------------------------------------
# Put it all together
def run(raw_dir, out_dir, max_files=None, target_attack=TARGET_PER_ATTACK,
     target_benign=TARGET_BENIGN, margin=MARGIN):
 files = sorted(Path(raw_dir).glob("*.csv"))   # sorted = same order every run
 if max_files:
     files = files[:max_files]                 # quick test: first N files only
 if not files:
     raise FileNotFoundError(f"No csv files found in {raw_dir}")

 rng = np.random.default_rng(SEED)             # created ONCE, used for all chunks
 # step 1: count rows per category and compute the keep chances
 counts = count_per_category(files)
 print("Rows per category in the raw data:", counts)

 targets = make_targets(target_attack, target_benign)
 keep_probs = make_keep_probs(counts, targets, margin)

 # steps 2-4: sample, clean, trim
 df = load_sample(files, keep_probs, rng)
 df, clean_info = clean(df)
 df, shortfalls = trim_to_target(df, targets)

 # steps 5-6: make labels, then split and scale
 X, y, cat = make_labels(df)
 arrays, scaler = split_and_scale(X, y, cat)

 # numbers for the report
 summary = {
     "seed": SEED,
     "files_read": len(files),
     "raw_rows_per_category": counts,
     "cleaning": clean_info,
     "shortfalls": shortfalls,
     "n_features": X.shape[1],
     "n_total": len(df),
     "n_train": len(arrays["y_train"]),
     "n_test": len(arrays["y_test"]),
 }
 # step 7: save everything
 save_outputs(out_dir, arrays, scaler, X.columns, summary)

 print(f"Features: {X.shape[1]} | Train rows: {summary['n_train']} "
       f"| Test rows: {summary['n_test']}")
 return arrays


# runs only when the file is executed directly, not when it is imported
if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="Preprocess CICIoT2023.")
 # folders for the input and output files
 parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
 parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
 parser.add_argument("--max-files", type=int, default=None,
                     help="only use the first N files (quick test)")
 parser.add_argument("--per-category", type=int, default=TARGET_PER_ATTACK)
 parser.add_argument("--benign", type=int, default=TARGET_BENIGN)
 parser.add_argument("--margin", type=float, default=MARGIN)
 args = parser.parse_args()
 # pass the options to the pipeline
 run(args.raw_dir, args.out_dir, args.max_files,
     args.per_category, args.benign, args.margin)
