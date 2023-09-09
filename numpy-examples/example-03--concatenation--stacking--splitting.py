import numpy as np

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = np.array([99, 99, 99])
# Concatenate one-dimensional arrays:
np.concatenate([x, y])
# array([1, 2, 3, 3, 2, 1])
np.concatenate([x, y, z])
# array([ 1,  2,  3,  3,  2,  1, 99, 99, 99])


grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
# Concatenate two-dimensional arrays:
np.concatenate([grid, grid])  # concatenate along the first axis (as additional rows)
# array([[1, 2, 3],
#        [4, 5, 6],
#        [1, 2, 3],
#        [4, 5, 6]])
np.concatenate([grid, grid], axis=1)  # concatenate along the second axis (zero-indexed) (as additional columns)
# array([[1, 2, 3, 1, 2, 3],
#        [4, 5, 6, 4, 5, 6]])

# Stack in axis 0 by np.vstack and in axis 1 by np.hstack:
np.vstack([x, grid])  # vertically stack the arrays
# array([[1, 2, 3],
#        [1, 2, 3],
#        [4, 5, 6]])
np.hstack([x[:2][:, np.newaxis], grid])
# array([[1, 1, 2, 3],
#        [2, 4, 5, 6]])

# Similarly, np.dstack for axis=2 (the third axis)

# Split a one-dimensional array by split points:
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])  # split points
# x1 is
# array([1, 2, 3])
# x2 is
# array([99, 99])
# x3 is
# array([3, 2, 1])

# Split by np.hsplit and np.vsplit:
grid = np.arange(16).reshape((4, 4))
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15]])
upper, lower = np.vsplit(grid, [2])
# for upper:
# array([[0, 1, 2, 3],
#        [4, 5, 6, 7]])
# for lower:
# array([[ 8,  9, 10, 11],
#        [12, 13, 14, 15]])
left, right = np.hsplit(grid, [2])
# left is
# array([[ 0,  1],
#        [ 4,  5],
#        [ 8,  9],
#        [12, 13]])
# right is
# array([[ 2,  3],
#        [ 6,  7],
#        [10, 11],
#        [14, 15]])

# Similarly np.dsplit for axis=2 (the third axis)
