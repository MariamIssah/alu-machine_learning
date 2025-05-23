Thanks! Here's a comprehensive `README.md` for **all the tasks** in your `math/linear_algebra` directory (basic + advanced), part of your `alu-machine_learning` repository.

---

# Linear Algebra Tasks

This directory contains Python implementations of various linear algebra tasks for the ALU Machine Learning curriculum. The tasks range from basic matrix operations to more advanced manipulations like slicing and concatenation.

---

## 📁 Files Overview

### `0-slice_me_up.py`

Slices a 2D matrix (list of lists) and prints specified submatrices.

#### Example Usage

```bash
$ ./0-slice_me_up.py
```

---

### `1-trim_me_down.py`

Defines a function `def trim_matrix(matrix, dim0, dim1):` that trims a matrix to the specified number of rows and columns.

#### Parameters

* `matrix`: 2D list
* `dim0`: int, number of rows to keep
* `dim1`: int, number of columns to keep

#### Returns

Trimmed 2D matrix.

---

### `2-size_me_please.py`

Defines a function `def matrix_shape(matrix):` that returns the shape (dimensions) of a matrix.

#### Parameters

* `matrix`: Nested list (can be any depth)

#### Returns

* A list of dimensions (e.g., `[2, 3]` for a 2x3 matrix)

---

### `3-flip_me_over.py`

Defines a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix.

#### Returns

* Transposed matrix.

---

### `4-line_up.py`

Defines a function `def add_arrays(arr1, arr2):` that adds two 1D arrays element-wise.

#### Returns

* A new list representing the sum.
* `None` if lengths mismatch.

---

### `5-across_the_planes.py`

Defines `def add_matrices2D(mat1, mat2):` that adds two 2D matrices element-wise.

#### Returns

* New matrix of the same dimensions.
* `None` if shapes mismatch.

---

### `6-howdy_partner.py`

Defines `def cat_arrays(arr1, arr2):` that concatenates two 1D arrays.

#### Returns

* Combined array as a list.

---

### `7-gettin_cozy.py`

Defines `def cat_matrices2D(mat1, mat2, axis=0):` that concatenates two 2D matrices along a given axis.

#### Returns

* Concatenated matrix.
* `None` if dimensions mismatch.

---

### `100-slice_like_a_ninja.py` *(Advanced Task)*

Defines `def np_slice(matrix, axes={}):` that slices a NumPy array along specified axes.

#### Example

```python
np_slice(matrix, axes={1: (1, 3)})
```

---

### `101-the_whole_barn.py` *(Advanced Task)*

Defines `def add_matrices(mat1, mat2):` that recursively adds matrices of any shape.

#### Returns

* Sum of both matrices.
* `None` if shapes mismatch.

---

### `102-squashed_like_sardines.py` *(Advanced Task)*

Defines `def cat_matrices(mat1, mat2, axis=0):` that recursively concatenates two matrices.

#### Returns

* Concatenated matrix.
* `None` if dimensions mismatch.

---

## 🧰 Requirements

* Python 3.x
* NumPy (for `100-slice_like_a_ninja.py` only)

## 🚀 Running the Tasks

You can run each task by executing its respective `main.py` file (e.g., `0-main.py`, `100-main.py`):

```bash
$ ./0-main.py
```

Ensure the task files have execution permissions:

```bash
$ chmod +x *.py
```

---

## 🧑‍💻 Author

**Mariam Awini Issah**
African Leadership University

---

## 📜 License

These scripts are part of the ALU Machine Learning curriculum and are intended for educational use.

---

Let me know if you’d like me to auto-generate this as a `.md` or `.txt` file!
