import numpy as np

# random generator
rng = np.random.default_rng(seed=42)  # seed for reproducibility (if you exit and restart your Python interpreter)

# Create:
# array([0, 7, 6, 4, 4, 8], dtype=int64)
x1 = rng.integers(10, size=6)  # one-dimensional array
print(x1[0])  # 0
print(x1[1])  # 7
print(x1[-1])  # 8
print(x1[-2])  # 4
x1[0] = 3.14159  # This will be truncated
# array([3, 7, 6, 4, 4, 8], dtype=int64)

# Create:
# array([[0, 6, 2, 0],
#        [5, 9, 7, 7],
#        [7, 7, 5, 1]], dtype=int64)
x2 = rng.integers(10, size=(3, 4))  # two-dimensional array
print(x2[0, 0])  # 0
print(x2[0, 1])  # 6
print(x2[1, 0])  # 5
print(x2[1, 1])  # 9
x2[0, 0] = 12
# array([[12,  6,  2,  0],
#        [ 5,  9,  7,  7],
#        [ 7,  7,  5,  1]], dtype=int64)
x2[0, 0] = 0
# array([[0, 6, 2, 0],
#        [5, 9, 7, 7],
#        [7, 7, 5, 1]], dtype=int64)

# Create:
# array([[[8, 4, 5, 3, 1],
#         [9, 7, 6, 4, 8],
#         [5, 4, 4, 2, 0],
#         [5, 8, 0, 8, 8]],
#
#        [[2, 6, 1, 7, 7],
#         [3, 0, 9, 4, 8],
#         [6, 7, 7, 1, 3],
#         [4, 4, 0, 5, 1]],
#
#        [[7, 6, 9, 7, 3],
#         [9, 4, 3, 9, 3],
#         [0, 4, 7, 1, 4],
#         [1, 6, 4, 3, 2]]], dtype=int64)
x3 = rng.integers(10, size=(3, 4, 5))  # three-dimensional array

# Slice to views:
# array([0, 7, 6], dtype=int64)
x1[:3]  # first three elements
# Slice to views:
# array([4, 4, 8], dtype=int64)
x1[3:]  # elements after index 3
# array([7, 6, 4], dtype=int64)
# Slice to views:
x1[1:4]  # middle subarray
# array([0, 6, 4], dtype=int64)
# Slice to views:
x1[::2]  # every second element
# array([8, 4, 4, 6, 7, 0], dtype=int64)
# Slice to views:
x1[::-1]  # all elements, reversed
# Slice to views:
# array([4, 6, 0], dtype=int64)
x1[4::-2]  # every second element from index 4, reversed

# Slice to views:
# array([[0, 6, 2],
#        [5, 9, 7]], dtype=int64)
x2[:2, :3]  # first two rows & three columns
# Slice to views:
# array([[1, 5, 7, 7],
#        [7, 7, 9, 5],
#        [0, 2, 6, 0]], dtype=int64)
x2[::-1, ::-1]  # all rows & columns, reversed
# Slice to views:
# array([0, 5, 7], dtype=int64)
x2[:, 0]  # first column of x2 as a one-dimensional array
# Slice to views:
# array([0, 6, 2, 0], dtype=int64)
x2[0, :]  # first row of x2
# Slice to views:
# array([0, 6, 2, 0], dtype=int64)
x2[0]  # equivalent to x2[0, :]

# Unlike Python list slices,
# NumPy array slices are returned as views
# rather than copies of the array data.

# Slice to views:
# array([[0, 6],
#        [5, 9]], dtype=int64)
x2_sub = x2[:2, :2]
x2_sub[0, 0] = 99
# for x2_sub:
# array([[99,  6],
#        [ 5,  9]], dtype=int64)
# for x2:
# array([[99,  6,  2,  0],
#        [ 5,  9,  7,  7],
#        [ 7,  7,  5,  1]], dtype=int64)

# Copy:
# array([[99,  6],
#        [ 5,  9]], dtype=int64)
x2_sub_copy = x2[:2, :2].copy()
x2_sub_copy[0, 0] = 42
# for x2_sub_copy:
# array([[42,  6],
#        [ 5,  9]], dtype=int64)
# for x2_sub:
# array([[99,  6],
#        [ 5,  9]], dtype=int64)

# Reshape to views:
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])
grid = np.arange(1, 10).reshape(3, 3)
# Reshape to views:
# array([[42,  2,  3,  4,  5,  6,  7,  8,  9]])
grid2 = grid.reshape(1, 9)  # row vector via reshape
grid2[0, 0] = 42
# for grid2:
# array([[42,  2,  3,  4,  5,  6,  7,  8,  9]])
# for grid:
# array([[42,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9]])
# Reshape to views:
# array([[42],
#        [ 2],
#        [ 3],
#        [ 4],
#        [ 5],
#        [ 6],
#        [ 7],
#        [ 8],
#        [ 9]])
grid3 = grid.reshape(9, 1)  # column vector via reshape

# A shorthand:
x = np.array([1, 2, 3])
# Reshape to the row vector:
# array([[1, 2, 3]])
x[np.newaxis, :]  # row vector via newaxis
# Reshape to the column vector:
# array([[1],
#        [2],
#        [3]])
x[:, np.newaxis]  # column vector via newaxis
