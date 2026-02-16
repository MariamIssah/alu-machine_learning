#!/usr/bin/env python3
"""
K-means clustering using scikit-learn.
"""

import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means on a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        k: number of clusters

    Returns:
        C: numpy.ndarray of shape (k, d) - centroid means
        clss: numpy.ndarray of shape (n,) - cluster index per point
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = model.cluster_centers_
    clss = model.labels_
    return C, clss
