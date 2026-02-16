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
        C_new = np.copy(C)
        empty = []
        j = 0
        while j < k:
            mask = clss == j
            if np.any(mask):
                C_new[j] = np.mean(X[mask], axis=0)
            else:
                empty.append(j)
            j += 1
        if empty:
            C_new[empty] = np.random.uniform(low=low, high=high,
                                             size=(len(empty), d))
        if np.allclose(C, C_new):
            return C_new, clss
        C = C_new
    return C, clss
