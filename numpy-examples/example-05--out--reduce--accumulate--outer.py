import numpy as np

x = np.arange(5)
# array([0, 1, 2, 3, 4])
y = np.empty(5)
np.multiply(x, 10, out=y)
# y is
# array([ 0., 10., 20., 30., 40.])

y = np.zeros(10)
np.power(2, x, out=y[::2])  # instead of y[::2] = 2 ** x to use memory efficiently
# y[::2] is
# array([ 1.,  2.,  4.,  8., 16.])
# y is
# array([ 1.,  0.,  2.,  0.,  4.,  0.,  8.,  0., 16.,  0.])

x = np.arange(1, 6)
# array([1, 2, 3, 4, 5])

# Reduce to:
np.add.reduce(x)  # 15
np.multiply.reduce(x)  # 120

x = np.arange(1, 6, 2)
# array([1, 3, 5])
# Reduce to:
# 1 / 3 / 5:
np.div.reduce(x)  # 0.06666666666666667

# Accumulate to:
np.add.accumulate(x)
# array([ 1,  3,  6, 10, 15])
np.multiply.accumulate(x)
# array([  1,   2,   6,  24, 120])

x = np.arange(1, 6)
# array([1, 2, 3, 4, 5])
np.multiply.outer(x, x)
# array([[ 1,  2,  3,  4,  5],
#        [ 2,  4,  6,  8, 10],
#        [ 3,  6,  9, 12, 15],
#        [ 4,  8, 12, 16, 20],
#        [ 5, 10, 15, 20, 25]])
