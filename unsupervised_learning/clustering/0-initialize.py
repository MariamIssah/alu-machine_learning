#!/usr/bin/env python3
"""
Initialize cluster centroids for K-means with multivariate uniform
distribution.
"""

import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        k: positive integer - number of clusters

    Returns:
        numpy.ndarray of shape (k, d) with initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids
