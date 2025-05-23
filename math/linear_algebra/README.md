# 🧮 Linear Algebra with Python

This repository contains a collection of Python scripts solving various linear algebra problems using lists and basic Python constructs. It is part of the ALU Machine Learning curriculum and is organized into basic and advanced tasks.

---

## 📁 Project Structure

Each Python file corresponds to a specific task in the study of linear algebra using pure Python or NumPy for advanced slicing. Below is a breakdown of what each script does, its functionality, and how to use it.

---

## 🟢 Basic Tasks

### `0-slice_me_up.py`

**Purpose**: Slice a 2D matrix and print specific submatrices.

**Example Output**:

```
First two rows:
[[1, 2, 3], [4, 5, 6]]
Last two rows:
[[7, 8, 9], [10, 11, 12]]
Middle two rows:
[[4, 5, 6], [7, 8, 9]]
```

---

### `1-trim_me_down.py`

**Function**: `trim_matrix(matrix, dim0, dim1)`

**Description**: Trims a matrix to the specified number of rows and columns.

**Returns**: A trimmed matrix of size `(dim0 x dim1)`.

---

### `2-size_me_please.py`

**Function**: `matrix_shape(matrix)`

**Description**: Returns the shape (dimensions) of a matrix.

**Returns**: A list indicating the size of each dimension, e.g., `[3, 2]`.

---

### `3-flip_me_over.py`

**Function**: `matrix_transpose(matrix)`

**Description**: Returns the transpose of a given 2D matrix.

---

### `4-line_up.py`

**Function**: `add_arrays(arr1, arr2)`

**Description**: Adds two 1D arrays element-wise.

**Returns**: A new list with summed elements or `None` if dimensions mismatch.

---

### `5-across_the_planes.py`

**Function**: `add_matrices2D(mat1, mat2)`

**Description**: Adds two 2D matrices element-wise.

**Returns**: A new matrix or `None` if shapes mismatch.

---

### `6-howdy_partner.py`

**Function**: `cat_arrays(arr1, arr2)`

**Description**: Concatenates two 1D arrays.

**Returns**: A new array.

---

### `7-gettin_cozy.py`

**Function**: `cat_matrices2D(mat1, mat2, axis=0)`

**Description**: Concatenates two 2D matrices along a specified axis.

**Returns**: Concatenated matrix or `None` if shapes mismatch.

---

## 🔴 Advanced Tasks

### `100-slice_like_a_ninja.py`

**Function**: `np_slice(matrix, axes={})`

**Description**: Performs slicing on a NumPy ndarray with flexible axis definitions.

**Returns**: A sliced NumPy array.

---

### `101-the_whole_barn.py`

**Function**: `add_matrices(mat1, mat2)`

**Description**: Recursively adds matrices of arbitrary depth.

**Returns**: A new matrix with the element-wise sum or `None` if shapes mismatch.

---

### `102-squashed_like_sardines.py`

**Function**: `cat_matrices(mat1, mat2, axis=0)`

**Description**: Recursively concatenates matrices of arbitrary depth along a specified axis.

**Returns**: A new concatenated matrix or `None` if shapes are incompatible.

---

## 🧪 Testing and Usage

Each script can be run independently. Example test scripts (like `0-main.py`, `100-main.py`, etc.) are provided for demonstration.

To run:

```bash
$ python3 0-main.py
```

Ensure NumPy is installed for advanced slicing:

```bash
$ pip install numpy
```

---

## ✍️ Author

**Mariam Awini Issah**  
Student at African Leadership University  
Passionate about machine learning and Python development.

---

## 📜 License

This project is intended for educational use within the ALU Machine Learning curriculum. All code is written for demonstration and learning purposes.
