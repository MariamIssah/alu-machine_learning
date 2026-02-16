#!/usr/bin/env python3
"""
Initialize variables for a Gaussian Mixture Model.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize variables for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) - the data set
        k: positive integer - number of clusters

    Returns:
        (pi, m, S) or (None, None, None) on failure.
        pi: shape (k,) priors, initialized evenly
        m: shape (k, d) centroid means, from K-means
        S: shape (k, d, d) covariance matrices, identity
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None
    n, d = X.shape
    pi = np.ones(k) / k
    C, _ = kmeans(X, k)
    if C is None:
        return None, None, None
    m = C
    S = np.tile(np.eye(d), (k, 1, 1))
    return pi, m, S
