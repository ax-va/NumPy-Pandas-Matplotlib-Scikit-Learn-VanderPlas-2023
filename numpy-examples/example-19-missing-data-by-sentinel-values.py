import numpy as np

vals1 = np.array([1, None, 2, 3])
# array([1, None, 2, 3], dtype=object)
# dtype=object is the best common type representation that is inferred by NumPy.
# That leads to much more overhead at the Python level than the typically fast operations at the NumPy level.

# %timeit np.arange(1E6, dtype=int).sum()
# 3.01 ms ± 11.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# %timeit np.arange(1E6, dtype=object).sum()
# 61.5 ms ± 2.04 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Moreover, arithmetical operations are not supported for None:

# vals1.sum()
# TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'

# For this reason, Pandas does not use None as a sentinel in its numerical arrays

vals2 = np.array([1, np.nan, 3, 4])
# array([ 1., nan,  3.,  4.])

1 + np.nan
# nan

0 + np.nan
# nan

vals2.sum(), vals2.min(), vals2.max()
# (nan, nan, nan)

np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)
# (8.0, 1.0, 4.0)

# There is no equivalent NaN value for integers, strings, or other types.
