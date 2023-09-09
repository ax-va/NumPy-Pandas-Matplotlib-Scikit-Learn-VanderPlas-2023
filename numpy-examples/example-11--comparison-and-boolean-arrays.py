import numpy as np

x = np.array([1, 2, 3, 4, 5])

# Create Boolean arrays:
x < 3  # # array([ True, True, False, False, False])
np.less(x, 3)  # array([ True,  True, False, False, False])

x > 3  # array([False, False, False, True, True])

x <= 3  # array([ True, True, True, False, False])

x >= 3  # array([False, False, True, True, True])

x != 3  # array([ True, True, False, True, True])

x == 3  # array([False, False, True, False, False])

# element-wise comparison of two arrays:
(2 * x) == (x ** 2)  # array([False, True, False, False, False])
# equivalent:
np.equal(2 * x, x ** 2)  # array([False, True, False, False, False])

rng = np.random.default_rng(seed=42)
# Create:
x = rng.integers(10, size=(3, 4))
# array([[0, 7, 6, 4],
#        [4, 8, 0, 6],
#        [2, 0, 5, 9]])

# Get Boolean values of comparison with 5:
x >= 5
# array([[False,  True,  True, False],
#        [False,  True, False,  True],
#        [False, False,  True,  True]])

# Count the number of True entries in a Boolean array:
np.count_nonzero(x >= 5)  # 6

# Alternatively, count the number of True entries in a Boolean array:
np.sum(x >= 5)  # 6

# Count the number of True entries in a Boolean array in each row:
np.sum(x >= 5, axis=1)  # array([2, 2, 2])
# Count the number of True entries in a Boolean array in each column:
np.sum(x >= 5, axis=0)  # array([0, 2, 2, 2])

# Is there any value greater than 5?
np.any(x > 5)  # True

# Are there all values greater than 5?
np.all(x > 5)   # False

# Is there any value greater than 5 in each row?
np.any(x > 5, axis=1)  # array([ True,  True,  True])
# Is there any value greater than 5 in each column?
np.any(x > 5, axis=0)  # array([False,  True,  True,  True])

# Are there all values greater than 0 in each row?
np.all(x > 0, axis=1)  # array([False, False, False])
# Are there any values greater than 0 in each column?
np.all(x > 0, axis=0)  # array([False, False, False,  True])

# Boolean elementwise operators: &, |, ^, and ~
np.sum((x > 0) & (x < 5))  # 3
# equivalent by using unfuncs:
np.sum(np.bitwise_and(x > 0, x < 5))  # 3
# De Morgan’s law yields the same result:
np.sum(~((x <= 0) | (x >= 5)))  # 3

# masking operation
x[x > 5]  # array([7, 6, 8, 6, 9])

# Python Boolean operations:
# - non-bitwise Boolean operations by using keywords:
print(bool(42), bool(0))  # True False
print(bool(42 and 0))  # False
print(bool(42 or 0))  # True
# - bitwise Boolean operations
print(bin(42))  # 0b101010
print(bin(59))  # 0b111011
print(bin(42 & 59))  # 0b101010
print(bin(42 | 59))  # 0b111011

# NumPy Boolean elementwise operations:
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B  # array([ True,  True,  True, False,  True,  True])
# A or B
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
# (x > 0) and (x < 5)
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

x = np.arange(10)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
(x > 4) & (x < 8)
# array([False, False, False, False, False,  True,  True,  True, False, False])

# (x > 4) and (x < 8)
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
