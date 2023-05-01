import numpy as np

# Concatenate a one-dimensional array:
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
# array([1, 2, 3, 3, 2, 1])
np.concatenate([x, y])
z = np.array([99, 99, 99])
# array([ 1,  2,  3,  3,  2,  1, 99, 99, 99])
np.concatenate([x, y, z])

# Concatenate a two-dimensional array:
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
# array([[1, 2, 3],
#        [4, 5, 6],
#        [1, 2, 3],
#        [4, 5, 6]])
np.concatenate([grid, grid])  # concatenate along the first axis
# array([[1, 2, 3, 1, 2, 3],
#        [4, 5, 6, 4, 5, 6]])
np.concatenate([grid, grid], axis=1)  # concatenate along the second axis (zero-indexed)

# np.vstack and np.hstack:
# array([[1, 2, 3],
#        [1, 2, 3],
#        [4, 5, 6]])
np.vstack([x, grid])  # vertically stack the arrays
# array([[1, 1, 2, 3],
#        [2, 4, 5, 6]])
np.hstack([x[:2][:, np.newaxis], grid])

# Similarly, np.dstack for axis=2

# Split by split points:
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])  # split points
# for x1:
# array([1, 2, 3])
# for x2:
# array([99, 99])
# for x3:
# array([3, 2, 1])

# np.hsplit and np.vsplit:
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15]])
grid = np.arange(16).reshape((4, 4))
upper, lower = np.vsplit(grid, [2])
# for upper:
# array([[0, 1, 2, 3],
#        [4, 5, 6, 7]])
# for lower:
# array([[ 8,  9, 10, 11],
#        [12, 13, 14, 15]])
left, right = np.hsplit(grid, [2])
# for left:
# array([[ 0,  1],
#        [ 4,  5],
#        [ 8,  9],
#        [12, 13]])
# for right:
# array([[ 2,  3],
#        [ 6,  7],
#        [10, 11],
#        [14, 15]])

# Similarly np.dsplit for axis=2
