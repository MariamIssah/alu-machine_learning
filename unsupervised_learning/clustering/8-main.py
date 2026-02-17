#!/usr/bin/env python3

import numpy as np
BIC = __import__('9-BIC').BIC

if __name__ == "__main__":
    X = np.random.rand(100, 3)
    print(BIC(X, kmax=1))
    print(BIC(X, kmin=5, kmax=4))
    print(BIC(X, kmin=100))
    print(BIC(X, kmin=101))
    