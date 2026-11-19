import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defence import compute_neighbours, knn_flag, outlier_keep, defend


def make_data():
    # 2 well-separated groups: 100 Benign (0) around 0, 100 Attack (1) around 10
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(10, 1, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


def test_knn_flags_flipped_sample():
    # Flip one Benign row to Attack; its neighbours disagree, so k-NN should flag it
    X, y = make_data()
    y_p = y.copy()
    y_p[5] = 1
    nb = compute_neighbours(X, cache=None)
    assert knn_flag(y_p, nb)[5]


def test_clean_data_mostly_kept():
    # Clean data (no flips): the defence should keep almost all 200 rows
    X, y = make_data()
    kept = defend(X, y, compute_neighbours(X, cache=None))
    assert len(kept) >= 190


def test_outlier_removed():
    # Move one row far away; the outlier filter should remove it
    X, y = make_data()
    X[0] = [50, 50]
    keep = outlier_keep(X, y, np.ones(len(y), bool))
    assert not keep[0]


def test_never_relabels():
    # The defence only removes rows: every kept index must be an original row
    X, y = make_data()
    kept = defend(X, y, compute_neighbours(X, cache=None))
    assert set(kept) <= set(range(len(y)))
