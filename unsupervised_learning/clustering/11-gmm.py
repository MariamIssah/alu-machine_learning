#!/usr/bin/env python3
"""
Gaussian Mixture Model using scikit-learn.
"""

import numpy as np
import sklearn.mixture


def gmm(X, k):
    """
    Calculate a GMM from a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        k: number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) - cluster priors
        m: numpy.ndarray of shape (k, d) - centroid means
        S: numpy.ndarray of shape (k, d, d) - covariance matrices
        clss: numpy.ndarray of shape (n,) - cluster indices per point
        bic: BIC value of the fitted model (scalar)
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)
    return pi, m, S, clss, bic
