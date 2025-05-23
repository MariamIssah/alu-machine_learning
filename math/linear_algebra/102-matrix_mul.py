#!/usr/bin/env python3
"""
Performs matrix multiplication using numpy.
"""

import numpy as np


def lazy_matrix_mul(m_a, m_b):
    """
    Multiplies two matrices using numpy.matmul (no loops or conditionals).

    Args:
        m_a (numpy.ndarray): First matrix.
        m_b (numpy.ndarray): Second matrix.

    Returns:
        numpy.ndarray: Result of multiplying m_a by m_b.
    """
    return np.matmul(m_a, m_b)