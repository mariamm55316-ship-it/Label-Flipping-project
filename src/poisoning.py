"""Label-flipping poisoning: the ATTACKER in our project.

It changes some training labels from Attack (1) to Benign (0), so the model
learns that some attacks are normal traffic. Benign rows are never changed.

Two strategies:
    random   : flip attack rows from any category (pool = 72,800 rows)
    targeted : flip only one category, DDoS by default (pool = 10,400 rows)

The rate is a share of the POOL, not of the whole training set:
    random   5% / 10% / 20%  ->  3,640 /  7,280 / 14,560 flips
    targeted 5% / 10% / 20%  ->    520 /  1,040 /  2,080 flips

Equal-flip runs: strategy="random" with n_flips=520 / 1040 / 2080.
Why: then random and targeted change the SAME number of labels, so only
the strategy is different and the comparison is fair.
"""
from __future__ import annotations

import numpy as np

STRATEGIES = ("random", "targeted")    # the only two strategies we allow


# ---------------------------------------------------------------------------
# Step 1: find the pool (the rows that MAY be flipped)
# ---------------------------------------------------------------------------
def attack_pool(y_train, cat_train, strategy, target_category="DDoS"):
    """Return the row numbers that may be flipped for this strategy."""
    y = np.asarray(y_train)            # make sure we work with numpy arrays
    cat = np.asarray(cat_train)
    is_attack = y == 1                 # True for every attack row

    if strategy == "random":
        mask = is_attack               # random: any attack row

    elif strategy == "targeted":
        in_target = cat == target_category      # True for every DDoS row
        # the category must exist, otherwise there is nothing to flip
        if not in_target.any():
            raise ValueError(f"target_category {target_category!r} not found in cat_train")
        # every DDoS row should be labelled 1; if not, the data has a bug
        if (in_target & ~is_attack).any():
            raise ValueError(f"some {target_category!r} rows are labelled Benign; check the data")
        mask = in_target               # targeted: only DDoS rows

    else:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    # flatnonzero turns True/False into row numbers, e.g. [F, T, T] -> [1, 2]
    return np.flatnonzero(mask)


# ---------------------------------------------------------------------------
# Input checks
# Why: wrong input should stop with a clear error, not give silently wrong results
# ---------------------------------------------------------------------------
def _validate_inputs(y_train, cat_train, seed):
    if y_train.ndim != 1 or cat_train.ndim != 1:              # must be simple lists of values
        raise ValueError("y_train and cat_train must be 1-D")
    if y_train.shape[0] != cat_train.shape[0]:                # one category per label
        raise ValueError("y_train and cat_train must have the same length")
    if not np.issubdtype(y_train.dtype, np.integer):          # labels are whole numbers
        raise ValueError(f"y_train must be an integer array, got {y_train.dtype}")
    if not np.isin(y_train, (0, 1)).all():                    # only 0 and 1 allowed
        raise ValueError("y_train must contain only 0 (Benign) and 1 (Attack)")
    # seed=None would give different flips every run, so results could not be repeated
    if seed is None or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer (None would not be reproducible)")


# ---------------------------------------------------------------------------
# Step 2: how many labels to flip
# ---------------------------------------------------------------------------
def _number_of_flips(rate, n_flips, pool_size):
    if n_flips is not None:
        # an exact number was given (equal-flip runs): use it and ignore rate
        if isinstance(n_flips, bool) or not isinstance(n_flips, (int, np.integer)):
            raise ValueError("n_flips must be an integer")
        n = int(n_flips)
    else:
        # otherwise use the rate: a share of the pool, between 0 and 1
        if rate is None or not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate!r}")
        n = int(round(float(rate) * pool_size))     # e.g. 0.05 * 72,800 = 3,640
    # we cannot flip more rows than the pool has
    if not 0 <= n <= pool_size:
        raise ValueError(f"cannot flip {n} labels from a pool of {pool_size}")
    return n


# ---------------------------------------------------------------------------
# The main function
# ---------------------------------------------------------------------------
def poison_labels(y_train, cat_train, strategy, rate, seed,
                  target_category="DDoS", n_flips=None):
    """Flip some Attack labels to Benign.

    Returns
    -------
    y_poisoned : a COPY of y_train with the chosen labels set to 0
    flipped_indices : the row numbers that were flipped (sorted)
        Why return them: the defence uses them to count how many
        poisoned rows it caught.
    """
    y_train = np.asarray(y_train)
    cat_train = np.asarray(cat_train)
    _validate_inputs(y_train, cat_train, seed)                 # stop early if input is wrong

    pool = attack_pool(y_train, cat_train, strategy, target_category)   # which rows may flip
    n = _number_of_flips(rate, n_flips, pool.size)                       # how many to flip

    # Step 3: choose the rows. Shuffle the pool with the seed, take the first n.
    # Why the seed: the same seed always gives the same rows, so results repeat.
    # Why "shuffle then take the first n": with the same seed, the 5% rows are
    # also inside the 10% rows, and the 10% rows inside the 20% rows (nested),
    # so a higher rate means "the same poison plus more".
    rng = np.random.default_rng(int(seed))
    flipped_indices = np.sort(rng.permutation(pool)[:n]).astype(np.int64)

    # Step 4: copy, then flip.
    # Why copy: the original clean labels must stay untouched for the clean baseline.
    y_poisoned = y_train.copy()
    y_poisoned[flipped_indices] = 0        # Attack (1) -> Benign (0)
    return y_poisoned, flipped_indices


# ---------------------------------------------------------------------------
# Demo: runs only when you press Run on this file (not when it is imported).
# It loads the real training labels and prints what each poisoning run does.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"   # change if your files are elsewhere
    y = np.load(data_dir / "y_train.npy")
    cat = np.load(data_dir / "cat_train.npy", allow_pickle=True)

    print(f"Training rows: {len(y):,}  |  Attack rows: {(y == 1).sum():,}  |  DDoS rows: {(cat == 'DDoS').sum():,}\n")
    for strategy, rate, n_flips in [("random", 0.05, None), ("random", 0.10, None), ("random", 0.20, None),
                                    ("targeted", 0.05, None), ("targeted", 0.10, None), ("targeted", 0.20, None),
                                    ("random", None, 520), ("random", None, 1040), ("random", None, 2080)]:
        y_p, idx = poison_labels(y, cat, strategy, rate, seed=42, n_flips=n_flips)
        name = f"{strategy} {rate:.0%}" if n_flips is None else f"random, equal {n_flips}"
        flipped_cats = sorted(set(cat[idx]))
        print(f"{name:<22} flipped {len(idx):>6,}  | attacks left: {(y_p == 1).sum():>6,}  | categories hit: {', '.join(flipped_cats)}")