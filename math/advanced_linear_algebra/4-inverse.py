#!/usr/bin/env python3
"""
Calculates the inverse of a matrix
"""
from 0-determinant import determinant
from 3-adjugate import adjugate


def inverse(matrix):
    """
    Calculates the inverse of a square matrix.
    Returns None if the matrix is singular.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    inv = [[round(adj[i][j] / det, 10) for j in range(len(matrix))] for i in range(len(matrix))]
    return inv
