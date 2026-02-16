#!/usr/bin/env python3
"""
Expectation-Maximization algorithm for a Gaussian Mixture Model.
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform expectation maximization for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) - the data set
        k: positive integer - number of clusters
        iterations: positive integer - maximum number of iterations
        tol: non-negative float - tolerance for log likelihood (early stop)
        verbose: bool - print log likelihood every 10 iterations

    Returns:
        (pi, m, S, g, l) or (None, None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    l_prev = None
    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        if verbose and (i % 10 == 0 or i == 0):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))
        if l_prev is not None and np.abs(l - l_prev) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {:.5f}".format(i, l))
            return pi, m, S, g, l
        l_prev = l
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None
    g, l = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {:.5f}".format(iterations - 1, l))
    return pi, m, S, g, l
