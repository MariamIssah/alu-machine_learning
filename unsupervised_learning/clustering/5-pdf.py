#!/usr/bin/env python3
"""
Probability density function of a multivariate Gaussian.
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculate the PDF of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) - data points
        m: numpy.ndarray of shape (d,) - mean of the distribution
        S: numpy.ndarray of shape (d, d) - covariance matrix

    Returns:
        P: numpy.ndarray of shape (n,) PDF values, min 1e-300, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None
    # P(x) = (2*pi)^(-d/2) * |S|^(-1/2) * exp(-0.5 * (x-m)^T S^{-1} (x-m))
    # No numpy.diag or .diagonal
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return None
    S_inv = np.linalg.inv(S)
    diff = X - m
    # (x-m)^T S^{-1} (x-m) per point: diff @ S_inv @ diff.T; per row: diff @ S_inv * diff
    mahal = np.sum(diff @ S_inv * diff, axis=1)
    log_p = (-0.5 * (d * np.log(2 * np.pi) + logdet + mahal))
    P = np.exp(log_p)
    P = np.maximum(P, 1e-300)
    return P
