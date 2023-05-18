import numpy as np

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b  # array([5, 6, 7])
a + 5  # array([5, 6, 7])

M = np.ones((3, 3))
# array([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.]])

M + a
# array([[1., 2., 3.],
#        [1., 2., 3.],
#        [1., 2., 3.]])

a = np.arange(3)
# array([0, 1, 2])
b = np.arange(3)[:, np.newaxis]
# array([[0],
#        [1],
#        [2]])

a + b
# array([[0, 1, 2],
#        [1, 2, 3],
#        [2, 3, 4]])

# Broadcasting rules:
#
# Rule 1: If the two arrays differ in their number of dimensions, the shape of the one with
# fewer dimensions is padded with ones on its leading (left) side.
#
# Rule 2: If the shape of the two arrays does not match in any dimension, the array with
# shape equal to 1 in that dimension is stretched to match the other shape.
#
# Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

# Example 1

M = np.ones((2, 3))
# array([[1., 1., 1.],
#        [1., 1., 1.]])

a = np.arange(3)
# array([0, 1, 2])

M + a
# array([[1., 2., 3.],
#        [1., 2., 3.]])

# Rules:
# M.shape is (2, 3)
# a.shape is (3,)
# Rule 1:
# M.shape remains (2, 3)
# a.shape becomes (1, 3)
# Rule 2:
# M.shape remains (2, 3)
# a.shape becomes (2, 3)

# Example 2

a = np.arange(3).reshape((3, 1))
# array([[0],
#        [1],
#        [2]])
b = np.arange(3)
# array([0, 1, 2])

a + b
# array([[0, 1, 2],
#        [1, 2, 3],
#        [2, 3, 4]])

# Rules:
# a.shape is (3, 1)
# b.shape is (3,)
# Rule 1:
# a.shape remains (3, 1)
# b.shape becomes (1, 3)
# Rule 2:
# a.shape becomes (3, 3)
# b.shape becomes (3, 3)

# Example 3

M = np.ones((3, 2))
# array([[1., 1.],
#        [1., 1.],
#        [1., 1.]])
a = np.arange(3)
# array([0, 1, 2])

# M + a
# ValueError: operands could not be broadcast together with shapes (3,2) (3,)

# Rules:
# M.shape is (3, 2)
# a.shape is (3,)
# Rule 1:
# M.shape remains (3, 2)
# a.shape becomes (1, 3)
# Rule 2:
# M.shape remains (3, 2)
# a.shape becomes (3, 3)
# Rule 3:
# Two arrays are incompatible

# Possible solution is to reshape the array to
a[:, np.newaxis]
# array([[0],
#        [1],
#        [2]])
M + a[:, np.newaxis]
# array([[1., 1.],
#        [2., 2.],
#        [3., 3.]])

# Broadcasting for other binary ufuncs, e.g. log(exp(a) + exp(b))
np.logaddexp(M, a[:, np.newaxis])
# array([[1.31326169, 1.31326169],
#        [1.69314718, 1.69314718],
#        [2.31326169, 2.31326169]])
