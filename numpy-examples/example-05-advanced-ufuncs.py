import numpy as np

# array([0, 1, 2, 3, 4])
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
# for y:
# array([ 0., 10., 20., 30., 40.])

y = np.zeros(10)
np.power(2, x, out=y[::2])  # instead of y[::2] = 2 ** x to use memory efficiently
# for y[::2]:
# array([ 1.,  2.,  4.,  8., 16.])
# for y:
# array([ 1.,  0.,  2.,  0.,  4.,  0.,  8.,  0., 16.,  0.])

# array([1, 2, 3, 4, 5])
x = np.arange(1, 6)
# 15
np.add.reduce(x)
# 120
np.multiply.reduce(x)
# array([ 1,  3,  6, 10, 15])
np.add.accumulate(x)
# array([  1,   2,   6,  24, 120])
np.multiply.accumulate(x)

# array([1, 3, 5])
x = np.arange(1, 6, 2)
# 1 / 3 / 5:
# 0.06666666666666667
np.div.reduce(x)

# array([1, 2, 3, 4, 5])
x = np.arange(1, 6)
# array([[ 1,  2,  3,  4,  5],
#        [ 2,  4,  6,  8, 10],
#        [ 3,  6,  9, 12, 15],
#        [ 4,  8, 12, 16, 20],
#        [ 5, 10, 15, 20, 25]])
np.multiply.outer(x, x)

