#!/usr/bin/env python3
"""
Find optimum number of clusters by variance (elbow method).
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Test for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) - the data set
        kmin: positive integer - minimum number of clusters (inclusive)
        kmax: positive integer - maximum number of clusters (inclusive)
        iterations: positive integer - max iterations for K-means

    Returns:
        (results, d_vars) or (None, None) on failure.
        results: list of (C, clss) from K-means for each k
        d_vars: list of variance difference from smallest cluster size
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    n = X.shape[0]
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin < 1 or not isinstance(kmax, int) or kmax < kmin:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    results = []
    vars_list = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))
        vars_list.append(variance(X, C))
    vars_arr = np.array(vars_list)
    var_smallest_k = vars_arr[0]
    d_vars = (var_smallest_k - vars_arr).tolist()
    return results, d_vars
