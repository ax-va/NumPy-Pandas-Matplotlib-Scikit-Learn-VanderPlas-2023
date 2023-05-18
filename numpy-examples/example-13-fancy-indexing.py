import numpy as np

rng = np.random.default_rng(seed=42)
x = rng.integers(100, size=10)  # array([ 8, 77, 65, 43, 43, 85,  8, 69, 20,  9])

ind = [3, 7, 4]
x[ind]  # array([43, 69, 43])

ind = np.array([[3, 7],
                [4, 5]])
# array([[43, 69],
#        [43, 85]])
x[ind]

# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])
X = np.arange(12).reshape((3, 4))

row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]  # array([ 2,  5, 11])
# array([[ 2,  1,  3],
#        [ 6,  5,  7],
#        [10,  9, 11]])
X[row[:, np.newaxis], col]

# explanation by broadcasting:
# - shape of row[:, np.newaxis] is (3, 1)
# 0
# 1
# 2

# - shape of col is (3, )
# - after broadcasting rule 1: (1, 3)
# - after broadcasting rule 2: (3, 3) for both arrays

# 0  0  0
# 1  1  1
# 2  2  2

# 2  1  3
# 2  1  3
# 2  1  3

# X[0, 2]  X[0, 1]  X(0, 3]
# X[1, 2]  X[1, 1]  X[1, 3]
# X[2, 2]  X[2, 1]  X[2, 3]

# 2  1  3
# 6  5  7
# 10  9  11

# The return value reflects the broadcasted shape of the indices,

# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])
X = np.arange(12).reshape((3, 4))

# combined indexing:
#  array([10,  8,  9])
X[2, [2, 0, 1]]
# array([[ 6,  4,  5],
#        [10,  8,  9]])
X[1:, [2, 0, 1]]

# masking
mask = np.array([True, False, True, False])
# array([0, 2])
X[0, mask]
