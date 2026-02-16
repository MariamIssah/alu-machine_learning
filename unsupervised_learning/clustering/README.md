# Clustering

Unsupervised learning: K-means, Gaussian Mixture Models (GMM), and Hierarchical (Agglomerative) clustering.

## Tasks

| #   | File                  | Description                                                         |
| --- | --------------------- | ------------------------------------------------------------------- |
| 0   | `0-initialize.py`     | Initialize K-means centroids with multivariate uniform distribution |
| 1   | `1-kmeans.py`         | K-means clustering (from scratch)                                   |
| 2   | `2-variance.py`       | Total intra-cluster variance                                        |
| 3   | `3-optimum.py`        | Optimum k by variance (elbow method)                                |
| 4   | `4-initialize.py`     | Initialize GMM (pi, m, S)                                           |
| 5   | `5-pdf.py`            | Gaussian PDF                                                        |
| 6   | `6-expectation.py`    | EM expectation step for GMM                                         |
| 7   | `7-maximization.py`   | EM maximization step for GMM                                        |
| 8   | `8-EM.py`             | Full EM algorithm for GMM                                           |
| 9   | `9-BIC.py`            | Best k for GMM via Bayesian Information Criterion                   |
| 10  | `10-kmeans.py`        | K-means using sklearn                                               |
| 11  | `11-gmm.py`           | GMM using sklearn                                                   |
| 12  | `12-agglomerative.py` | Agglomerative clustering with Ward linkage (scipy)                  |

## Requirements

- Python 3.5
- NumPy 1.15, scikit-learn 0.21, SciPy 1.3

## Usage

Run main files from the project description (e.g. `./0-main.py`) or import functions:

```python
from importlib import import_module
initialize = __import__('0-initialize').initialize
kmeans = __import__('1-kmeans').kmeans
# ...
```
