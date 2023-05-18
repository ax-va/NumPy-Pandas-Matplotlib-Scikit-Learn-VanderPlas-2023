import numpy as np

# random generator
rng = np.random.default_rng(seed=42)  # seed for reproducibility (if you exit and restart your Python interpreter)

# Create a one-dimensional array (neither a row vector nor a column vector):
x1 = rng.integers(10, size=6)  # one-dimensional array
# array([0, 7, 6, 4, 4, 8])
print(x1[0])  # 0
print(x1[1])  # 7
print(x1[-1])  # 8
print(x1[-2])  # 4
x1[0] = 3.14159  # This will be truncated
# x1 is
# array([3, 7, 6, 4, 4, 8])

# Create a two-dimensional array:
x2 = rng.integers(10, size=(3, 4))  # two-dimensional array
# array([[0, 6, 2, 0],
#        [5, 9, 7, 7],
#        [7, 7, 5, 1]])
print(x2[0, 0])  # 0
print(x2[0, 1])  # 6
print(x2[1, 0])  # 5
print(x2[1, 1])  # 9
x2[0, 0] = 12
# array([[12,  6,  2,  0],
#        [ 5,  9,  7,  7],
#        [ 7,  7,  5,  1]])
x2[0, 0] = 0
# array([[0, 6, 2, 0],
#        [5, 9, 7, 7],
#        [7, 7, 5, 1]])

# Create a three-dimensional array:
x3 = rng.integers(10, size=(3, 4, 5))  # three-dimensional array
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
#         [1, 6, 4, 3, 2]]])

# x1 is
# array([3, 7, 6, 4, 4, 8])

# Slice to views:
x1[:3]  # first three elements
# array([3, 7, 6])
# Slice to views:
x1[3:]  # elements after index 3 (included)
# array([4, 4, 8])
# Slice to views:
x1[1:4]  # middle subarray
# array([7, 6, 4])
# Slice to views:
x1[::2]  # every second element
# array([3, 6, 4])
# Slice to views:
x1[::-1]  # all elements, reversed
# array([8, 4, 4, 6, 7, 3])
# Slice to views:
x1[4::-2]  # every second element from index 4, reversed
# array([4, 6, 3])

# x2 is
# array([[0, 6, 2, 0],
#        [5, 9, 7, 7],
#        [7, 7, 5, 1]])

# Slice to views:
x2[:2, :3]  # first two rows & three columns
# array([[0, 6, 2],
#        [5, 9, 7]])
# Slice to views:
x2[::-1, ::-1]  # all rows & columns, reversed
# array([[1, 5, 7, 7],
#        [7, 7, 9, 5],
#        [0, 2, 6, 0]])
# Slice to views:
x2[:, 0]  # first column of x2 as a one-dimensional array
# array([0, 5, 7])
# Slice to views:
x2[0, :]  # first row of x2
# array([0, 6, 2, 0])
# Slice to views:
x2[0]  # equivalent to x2[0, :]
# array([0, 6, 2, 0])

# Unlike Python list slices,
# NumPy array slices are returned as views
# rather than copies of the array data.

# Slice to views:
x2_sub = x2[:2, :2]
# array([[0, 6],
#        [5, 9]])
x2_sub[0, 0] = 99
# x2_sub is
# array([[99,  6],
#        [ 5,  9]], dtype=int64)
# x2 is
# array([[99,  6,  2,  0],
#        [ 5,  9,  7,  7],
#        [ 7,  7,  5,  1]], dtype=int64)

# Copy:

x2_sub_copy = x2[:2, :2].copy()
# array([[99,  6],
#        [ 5,  9]], dtype=int64)
x2_sub_copy[0, 0] = 42
# 2_sub_copy is
# array([[42,  6],
#        [ 5,  9]], dtype=int64)
# x2_sub is
# array([[99,  6],
#        [ 5,  9]], dtype=int64)

# Reshape to views:
grid = np.arange(1, 10).reshape(3, 3)
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])
# Reshape to views:
grid2 = grid.reshape(1, 9)  # row vector via reshape
# array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
grid2[0, 0] = 42
# grid2 is
# array([[42,  2,  3,  4,  5,  6,  7,  8,  9]])
# grid is
# array([[42,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9]])
# Reshape to views:
grid3 = grid.reshape(9, 1)  # column vector via reshape
# array([[42],
#        [ 2],
#        [ 3],
#        [ 4],
#        [ 5],
#        [ 6],
#        [ 7],
#        [ 8],
#        [ 9]])

# Shorthands to reshape to row and column vectors via np.newaxis:
x = np.array([1, 2, 3])
# Reshape to the row vector:
x[np.newaxis, :]  # row vector via np.newaxis
# array([[1, 2, 3]])
# Reshape to the column vector:
x[:, np.newaxis]  # column vector via np.newaxis
# array([[1],
#        [2],
#        [3]])
