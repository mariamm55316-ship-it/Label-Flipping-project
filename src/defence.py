"""Two-stage label sanitisation defence (owner: Haneen).

Uses only the (possibly poisoned) training data - no trusted clean data.
Stage 1: k-NN label check (k=10, flag if >= 7 neighbours disagree).
Stage 2: per-class centroid outlier filter (distance > mean + 3*std).
Flagged rows are removed, never relabelled.
"""
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_neighbours(X, k=10, cache="data/processed/neighbours.npy"):
    """Features never change between runs, so neighbours are computed once and cached."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X).kneighbors(X)[1]
    # remove each row itself by value (ties mean self is not always first)
    not_self = idx != np.arange(len(X))[:, None]
    neighbours = idx[not_self].reshape(-1, k)
    if cache:
        np.save(cache, neighbours)
    return neighbours


def knn_flag(y, neighbours, threshold=7):
    disagree = (y[neighbours] != y[:, None]).sum(axis=1)
    return disagree >= threshold


def outlier_keep(X, y, keep):
    keep = keep.copy()
    for cl in np.unique(y):
        rows = np.where(keep & (y == cl))[0]
        d = np.linalg.norm(X[rows] - X[rows].mean(axis=0), axis=1)
        keep[rows[d > d.mean() + 3 * d.std()]] = False
    return keep


def defend(X_train, y_poisoned, neighbours=None, stages="both"):
    """Return kept_indices. stages: 'knn', 'outlier' or 'both'."""
    keep = np.ones(len(y_poisoned), bool)
    if stages in ("knn", "both"):
        if neighbours is None:
            neighbours = compute_neighbours(X_train)
        keep &= ~knn_flag(y_poisoned, neighbours)
    if stages in ("outlier", "both"):
        keep = outlier_keep(X_train, y_poisoned, keep)
    return np.where(keep)[0]
