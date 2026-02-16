#!/usr/bin/env python3
"""
Find best number of clusters for a GMM using the Bayesian Information Criterion.
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) - the data set
        kmin: positive integer - minimum number of clusters (inclusive)
        kmax: positive integer - maximum number of clusters (inclusive)
        iterations: max iterations for EM
        tol: tolerance for EM
        verbose: bool - EM prints to stdout

    Returns:
        (best_k, best_result, l, b) or (None, None, None, None) on failure.
        best_result: tuple (pi, m, S)
        l: log likelihood for each k, shape (kmax - kmin + 1)
        b: BIC for each k, shape (kmax - kmin + 1)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin < 1 or not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None
    l_list = []
    b_list = []
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_lik = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose)
        if pi is None:
            return None, None, None, None
        results.append((pi, m, S))
        l_list.append(log_lik)
        # p = number of parameters: (k-1) + k*d + k*d*(d+1)/2
        p = (k - 1) + k * d + k * d * (d + 1) // 2
        bic = p * np.log(n) - 2 * log_lik
        b_list.append(bic)
    l_arr = np.array(l_list)
    b_arr = np.array(b_list)
    best_idx = np.argmin(b_arr)
    best_k = kmin + best_idx
    best_result = results[best_idx]
    return best_k, best_result, l_arr, b_arr
