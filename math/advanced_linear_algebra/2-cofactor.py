#!/usr/bin/env python3
"""
Calculates the cofactor matrix
"""
from 1-minor import minor


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minors = minor(matrix)
    for i in range(len(minors)):
        for j in range(len(minors)):
            minors[i][j] *= (-1) ** (i + j)
    return minors
