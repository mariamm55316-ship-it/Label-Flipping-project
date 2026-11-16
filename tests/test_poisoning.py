import sys
from pathlib import Path

import numpy as np
import pytest

# poisoning.py is in the src folder, so tell Python where to look
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poisoning import attack_pool, poison_labels


def make_data():
    # Small fake dataset: 500 Benign rows + 100 rows of each of 7 attack types
    # So: 700 attack rows (random pool) and 100 DDoS rows (targeted pool)
    attacks = ["DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
    cat = np.array(["Benign"] * 500 + [a for a in attacks for _ in range(100)])
    cat = cat[np.random.default_rng(0).permutation(cat.size)]   # shuffle
    y = (cat != "Benign").astype(np.int8)                        # 0 = Benign, 1 = Attack
    return y, cat


def test_random_flips_right_number():
    # 10% of 700 attack rows = 70 flips
    y, cat = make_data()
    y_p, idx = poison_labels(y, cat, "random", 0.10, seed=42)
    assert idx.size == 70
    assert (y != y_p).sum() == 70


def test_targeted_flips_right_number_and_only_ddos():
    # 10% of 100 DDoS rows = 10 flips, all of them DDoS
    y, cat = make_data()
    y_p, idx = poison_labels(y, cat, "targeted", 0.10, seed=42)
    assert idx.size == 10
    assert set(cat[idx]) == {"DDoS"}


def test_exact_number_of_flips():
    # n_flips gives an exact number (used for the equal-flip runs)
    y, cat = make_data()
    _, idx = poison_labels(y, cat, "random", None, seed=42, n_flips=20)
    assert idx.size == 20


def test_only_attack_becomes_benign():
    # flipped rows were Attack (1) and are now Benign (0); nothing else changed
    y, cat = make_data()
    y_p, idx = poison_labels(y, cat, "random", 0.20, seed=42)
    assert (y[idx] == 1).all()
    assert (y_p[idx] == 0).all()
    assert np.array_equal(np.flatnonzero(y != y_p), idx)


def test_zero_rate_changes_nothing():
    y, cat = make_data()
    y_p, idx = poison_labels(y, cat, "random", 0.0, seed=42)
    assert idx.size == 0
    assert np.array_equal(y, y_p)


def test_original_labels_not_changed():
    # the clean labels are still needed for the clean baseline
    y, cat = make_data()
    y_before = y.copy()
    poison_labels(y, cat, "random", 0.20, seed=42)
    assert np.array_equal(y, y_before)


def test_same_seed_same_rows():
    y, cat = make_data()
    _, a = poison_labels(y, cat, "random", 0.10, seed=7)
    _, b = poison_labels(y, cat, "random", 0.10, seed=7)
    assert np.array_equal(a, b)


def test_small_rate_rows_inside_bigger_rate_rows():
    # same seed: the 5% rows are part of the 10% rows, which are part of the 20% rows
    y, cat = make_data()
    s05 = set(poison_labels(y, cat, "random", 0.05, seed=42)[1])
    s10 = set(poison_labels(y, cat, "random", 0.10, seed=42)[1])
    s20 = set(poison_labels(y, cat, "random", 0.20, seed=42)[1])
    assert s05 <= s10 <= s20


def test_bad_input_gives_error():
    y, cat = make_data()
    with pytest.raises(ValueError):
        poison_labels(y, cat, "label_swap", 0.10, seed=42)   # unknown strategy
    with pytest.raises(ValueError):
        poison_labels(y, cat, "random", 1.5, seed=42)        # rate above 100%
    with pytest.raises(ValueError):
        poison_labels(y, cat, "random", 0.10, seed=None)     # no seed


# Real CICIoT2023 training data (skipped if data/processed is missing)
DATA_DIR = Path("data/processed")
HAVE_DATA = (DATA_DIR / "y_train.npy").exists() and (DATA_DIR / "cat_train.npy").exists()


@pytest.mark.skipif(not HAVE_DATA, reason="processed data not found")
def test_real_data_numbers_match_report():
    y = np.load(DATA_DIR / "y_train.npy")
    cat = np.load(DATA_DIR / "cat_train.npy", allow_pickle=True)
    assert attack_pool(y, cat, "random").size == 72800     # all attack rows
    assert attack_pool(y, cat, "targeted").size == 10400   # all DDoS rows
    _, idx = poison_labels(y, cat, "random", 0.10, seed=42)
    assert idx.size == 7280
    _, idx = poison_labels(y, cat, "targeted", 0.10, seed=42)
    assert idx.size == 1040
