#!/usr/bin/env python3
"""
K-means clustering implementation.
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        k: positive integer - number of clusters
        iterations: positive integer - maximum number of iterations

    Returns:
        (C, clss) or (None, None) on failure.
        C: centroids shape (k, d), clss: cluster index per point shape (n,)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low=low, high=high, size=(k, d))
    for _ in range(iterations):
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        clss = np.argmin(dist_sq, axis=1)
        C_new = np.zeros((k, d), dtype=X.dtype)
        np.add.at(C_new, clss, X)
        n_per = np.bincount(clss, minlength=k)
        np.divide(C_new, n_per[:, np.newaxis], out=C_new,
                  where=n_per[:, np.newaxis] > 0)
        empty = (n_per == 0)
        if np.any(empty):
            C_new[empty] = np.random.uniform(low=low, high=high,
                                             size=(np.sum(empty), d))
        if np.allclose(C, C_new):
            return C_new, clss
        C = C_new
    return C, clss
