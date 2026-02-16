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
        # Assign each point to nearest centroid (no loop: use broadcasting)
        # distances shape (n, k): each point's distance to each centroid
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]  # (n, k, d)
        dist_sq = np.sum(diff * diff, axis=2)  # (n, k)
        clss = np.argmin(dist_sq, axis=1)  # (n,)
        # Update centroids: vectorized only (no second loop)
        one_hot = (clss[:, np.newaxis] == np.arange(k)).astype(np.float64)
        n_per = np.sum(one_hot, axis=0)
        safe_n = np.where(n_per > 0, n_per, 1)[:, np.newaxis]
        C_new = (one_hot.T @ X) / safe_n
        empty_mask = (n_per == 0)
        if np.any(empty_mask):
            n_empty = int(np.sum(empty_mask))
            C_new[empty_mask] = np.random.uniform(low=low, high=high,
                                                  size=(n_empty, d))
        if np.allclose(C, C_new):
            return C_new, clss
        C = C_new
    return C, clss
