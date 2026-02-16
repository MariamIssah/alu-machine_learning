#!/usr/bin/env python3
"""
Agglomerative clustering with Ward linkage and dendrogram.
"""

import numpy as np
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Perform agglomerative clustering with Ward linkage.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        dist: maximum cophenetic distance for all clusters

    Returns:
        clss: numpy.ndarray of shape (n,) - cluster indices for each point
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist, criterion='distance')
    dendrogram = scipy.cluster.hierarchy.dendrogram(
        linkage, color_threshold=dist, above_threshold_color='gray')
    plt.show()
    return clss
