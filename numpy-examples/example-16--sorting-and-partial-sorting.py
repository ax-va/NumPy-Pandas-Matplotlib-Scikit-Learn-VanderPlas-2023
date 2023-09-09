import numpy as np

x = np.array([2, 1, 4, 3, 5])
# Return a sorted copy
x_sorted = np.sort(x)
# x_sorted is
# array([1, 2, 3, 4, 5])
# x is
# array([2, 1, 4, 3, 5])

# Sort in place
x.sort()
# x is
# array([1, 2, 3, 4, 5])

x = np.array([2, 1, 4, 3, 5])
# Return the indices of the sorted elements
i = np.argsort(x)
# i is
# array([1, 0, 3, 2, 4])
# x is
# array([2, 1, 4, 3, 5])

# fancy indexing
x[i]
# array([1, 2, 3, 4, 5])

rng = np.random.default_rng(seed=42)
X = rng.integers(0, 10, (4, 6))
# array([[0, 7, 6, 4, 4, 8],
#        [0, 6, 2, 0, 5, 9],
#        [7, 7, 7, 7, 5, 1],
#        [8, 4, 5, 3, 1, 9]])

# Sort columns
np.sort(X, axis=0)
# array([[0, 4, 2, 0, 1, 1],
#        [0, 6, 5, 3, 4, 8],
#        [7, 7, 6, 4, 5, 9],
#        [8, 7, 7, 7, 5, 9]])
# Sort rows
np.sort(X, axis=1)
# array([[0, 4, 4, 6, 7, 8],
#        [0, 0, 2, 5, 6, 9],
#        [1, 5, 7, 7, 7, 7],
#        [1, 3, 4, 5, 8, 9]])

x = np.array([7, 2, 3, 1, 6, 5, 4])
# Return 3 smallest numbers to the left and the remaining numbers to the right
np.partition(x, 3)
# array([2, 1, 3, 4, 6, 5, 7])
# Within the two partitions, the elements have arbitrary order

np.partition(X, 2, axis=1)
# array([[0, 4, 4, 7, 6, 8],
#        [0, 0, 2, 6, 5, 9],
#        [1, 5, 7, 7, 7, 7],
#        [1, 3, 4, 5, 8, 9]])

# First two slots in each row contain the smallest values from that row

# Similarly np.argpartition
