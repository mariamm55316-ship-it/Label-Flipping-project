"""Evaluation metrics. Attack = 1, Benign = 0. Rates are fractions.
"""
from numbers import Integral
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score


def _indices(values, n_train, name):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        return np.array([], dtype=np.int64)
    if values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer row indices")
    if np.any(values < 0) or np.any(values >= n_train):
        raise ValueError(f"{name} contains an out-of-range row index")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} contains duplicate row indices")
    return values.astype(np.int64, copy=False)


def defence_metrics(n_train, flipped_indices, kept_indices):
    """Evaluate removed samples against the recorded label flips."""
    if isinstance(n_train, bool) or not isinstance(n_train, Integral) or n_train < 0:
        raise ValueError("n_train must be a non-negative integer")
    flipped_indices = _indices(flipped_indices, n_train, "flipped_indices")
    kept_indices = _indices(kept_indices, n_train, "kept_indices")
    removed_indices = np.setdiff1d(np.arange(n_train), kept_indices)
    caught_indices = np.intersect1d(removed_indices, flipped_indices)
    flipped = len(flipped_indices)
    removed = len(removed_indices)
    caught = len(caught_indices)
    unpoisoned = n_train - flipped
    return {
        "n_flipped": flipped,
        "n_removed": removed,
        "n_caught": caught,
        "poison_detection_rate": caught / flipped if flipped else np.nan,
        "removal_precision": caught / removed if removed else np.nan,
        "clean_removal_rate": (removed - caught) / unpoisoned if unpoisoned else np.nan,
    }


def _binary_pair(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or not len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Labels and predictions must be non-empty 1-D arrays of matching length")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Labels and predictions must contain only 0 or 1")
    return y_true, y_pred


def detection_metrics(y_true, y_pred):
    """Calculate binary classification metrics."""
    y_true, y_pred = _binary_pair(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(tp / (tp + fn)) if tp + fn else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else np.nan,
    }


def per_category_recall(y_true, y_pred, cat_test):
    """Calculate recall for each attack category."""
    y_true, y_pred = _binary_pair(y_true, y_pred)
    cat_test = np.asarray(cat_test)
    if cat_test.ndim != 1 or len(cat_test) != len(y_true):
        raise ValueError("Test categories must match the label row order and length")
    if not np.array_equal(y_true == 0, cat_test == "Benign"):
        raise ValueError("Test categories do not agree with the binary labels")
    return {
        "recall_" + str(category): float((y_pred[cat_test == category] == 1).mean())
        for category in sorted(set(cat_test)) if category != "Benign"
    }


def attack_success_rate(y_true, y_pred, cat_test, target_category="DDoS"):
    """Return the target miss rate: 1 - target recall; not baseline-adjusted."""
    recalls = per_category_recall(y_true, y_pred, cat_test)
    return 1.0 - recalls.get("recall_" + target_category, np.nan)


def classification_metrics(y_true, y_pred, cat_test):
    """Combine detection and category recall metrics."""
    return {**detection_metrics(y_true, y_pred),
            **per_category_recall(y_true, y_pred, cat_test)}
