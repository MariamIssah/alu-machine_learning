#!/usr/bin/env python3
"""
Total intra-cluster variance for K-means.
"""

import numpy as np


def variance(X, C):
    """
    Calculate the total intra-cluster variance for a data set.

    Args:
        X: numpy.ndarray of shape (n, d) - the data set
        C: numpy.ndarray of shape (k, d) - centroid means for each cluster

    Returns:
        Total variance (float), or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    n, d = X.shape
    k = C.shape[0]
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    clss = np.argmin(dist_sq, axis=1)
    centroids_for_points = C[clss]
    var = np.sum((X - centroids_for_points) ** 2)
    return var
