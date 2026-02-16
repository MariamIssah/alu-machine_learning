#!/usr/bin/env python3
"""
Expectation step in the EM algorithm for a GMM.
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) - the data set
        pi: numpy.ndarray of shape (k,) - priors for each cluster
        m: numpy.ndarray of shape (k, d) - centroid means
        S: numpy.ndarray of shape (k, d, d) - covariance matrices

    Returns:
        (g, l) or (None, None) on failure.
        g: shape (k, n) posterior probabilities
        l: total log likelihood
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or not isinstance(m, np.ndarray):
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    weighted = np.zeros((k, n))
    for j in range(k):
        P_j = pdf(X, m[j], S[j])
        if P_j is None:
            return None, None
        weighted[j] = pi[j] * P_j
    total = np.sum(weighted, axis=0, keepdims=True)
    g = np.where(total > 0, weighted / total, 0)
    l = np.sum(np.log(np.sum(weighted, axis=0) + 1e-300))
    return g, l
