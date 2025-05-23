#!/usr/bin/env python3
import numpy as np

def np_slice(matrix, axes={}):
    slices = [slice(None)] * matrix.ndim
    for axis, sl in axes.items():
        slices[axis] = slice(*sl)
    return matrix[tuple(slices)]