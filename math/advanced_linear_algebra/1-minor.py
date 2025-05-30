#!/usr/bin/env python3
"""
Calculates the minor matrix of a matrix
"""
from 0-determinant import determinant


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minors = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            # Get submatrix
            submatrix = [r[:j] + r[j+1:] for idx, r in enumerate(matrix) if idx != i]
            row.append(determinant(submatrix))
        minors.append(row)
    return minors
