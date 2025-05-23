#!/usr/bin/env python3
from copy import deepcopy

def cat_matrices(mat1, mat2, axis=0):
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if axis == 0:
        if mat1 and mat2 and type(mat1[0]) != type(mat2[0]):
            return None
        return deepcopy(mat1) + deepcopy(mat2)
    if len(mat1) != len(mat2):
        return None
    result = []
    for sub1, sub2 in zip(mat1, mat2):
        res = cat_matrices(sub1, sub2, axis=axis - 1)
        if res is None:
            return None
        result.append(res)
    return result
