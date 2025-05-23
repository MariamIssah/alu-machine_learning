#!/usr/bin/env python3

def add_matrices(mat1, mat2):
    if type(mat1) != list or type(mat2) != list:
        return None
    if len(mat1) != len(mat2):
        return None
    if isinstance(mat1[0], list):
        return [add_matrices(sub1, sub2) for sub1, sub2 in zip(mat1, mat2)]
    return [a + b for a, b in zip(mat1, mat2)]
