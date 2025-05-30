#!/usr/bin/env python3
"""
Calculates the adjugate matrix
"""
from 2-cofactor import cofactor


def adjugate(matrix):
    """
    Returns the adjugate (transposed cofactor) of a matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cof = cofactor(matrix)
    # Transpose the cofactor matrix
    adj = [[cof[j][i] for j in range(len(cof))] for i in range(len(cof))]
    return adj
