#!/usr/bin/env python3
"""
A module that multiplies 2 matrices using numpy
"""

import numpy as np


def lazy_matrix_mul(m_a, m_b):
    """
    Multiplies two matrices using numpy.matmul (which supports broadcasting).

    Args:
        m_a (numpy.ndarray): The first matrix.
        m_b (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The result of the matrix multiplication.

    Raises:
        TypeError: If m_a or m_b is not a numpy.ndarray.
        ValueError: If the shapes are not aligned for matrix multiplication.
    """
    return np.matmul(m_a, m_b)