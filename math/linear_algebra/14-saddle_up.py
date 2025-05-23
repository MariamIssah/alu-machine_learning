#!/usr/bin/env python3
"""
A module that multiplies 2 matrices using numpy
"""

import numpy as np

def np_matmul(m_a, m_b):
    """
    Multiplies two matrices using numpy.matmul.

    Args:
        m_a (numpy.ndarray): The first matrix.
        m_b (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The result of the matrix multiplication.
    """
    return np.matmul(m_a, m_b)
