import numpy as np
import pandas as pd
import numexpr

# motivation

rng = np.random.default_rng(42)
x = rng.random(1000000)
y = rng.random(1000000)

# very efficient:
# %timeit x + y
# 3.32 ms ± 440 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# not efficient:
# %timeit np.fromiter((xi + yi for xi, yi in zip(x, y)), dtype=x.dtype, count=len(x))
# 261 ms ± 403 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

# less efficient:
mask = (x > 0.5) & (y < 0.5)
# Because it is roughly equivalent to compound expressions
# and every intermediate step is explicitly allocated in memory.
tmp1 = (x > 0.5)
tmp2 = (y < 0.5)
mask = tmp1 & tmp2

# without allocating full intermediate arrays in memory => much more efficient than NumPy
mask_numexpr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')
np.all(mask == mask_numexpr)
# True

# pandas.eval for efficient operations

nrows, ncols = 100000, 100
df1, df2, df3, df4 = (pd.DataFrame(rng.random((nrows, ncols))) for i in range(4))

# typical Pandas approach:
# %timeit df1 + df2 + df3 + df4

# behaviour in IPython on Ubuntu

# before installing NumExpr

# 97.6 ms ± 13.8 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 89 ms ± 375 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 87.5 ms ± 104 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 87.5 ms ± 968 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# after installing NumExpr

# 264 ms ± 2.95 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 265 ms ± 2.71 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 265 ms ± 2.35 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 265 ms ± 2.59 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# after deleting NumExpr

# 86.4 ms ± 154 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 86.9 ms ± 197 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# %timeit pd.eval('df1 + df2 + df3 + df4')

# before installing NumExpr

# 98.2 ms ± 14.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 88.8 ms ± 183 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 88.8 ms ± 351 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 88 ms ± 177 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# after installing NumExpr

# 97.4 ms ± 545 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 98.3 ms ± 1.6 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 98.8 ms ± 1.3 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# after deleting NumExpr

# 88.3 ms ± 777 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 87.7 ms ± 375 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# the same result
np.allclose(df1 + df2 + df3 + df4, pd.eval('df1 + df2 + df3 + df4'))
# True

df1, df2, df3, df4, df5 = (pd.DataFrame(rng.integers(0, 1000, (100, 3))) for i in range(5))
df1
#       0    1    2
# 0   595  712  318
# 1    23  563  586
# 2   701  679  458
# 3   167  527  737
# 4   621   33  440
# ..  ...  ...  ...
# 95  460  412  291
# 96  702  603  717
# 97  548  971  129
# 98   59  802  395
# 99  844    5  447
#
# [100 rows x 3 columns]

# arithmetic operators
result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2)
# True

# comparison operators
result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
#         0      1      2
# 0   False  False  False
# 1    True  False  False
# 2   False  False  False
# 3   False  False  False
# 4   False  False  False
# ..    ...    ...    ...
# 95   True   True  False
# 96  False  False  False
# 97  False  False   True
# 98  False  False  False
# 99  False   True  False
#
# [100 rows x 3 columns]
result2 = pd.eval('df1 < df2 <= df3 != df4')
np.allclose(result1, result2)
# True
result3 = pd.eval('(df1 < df2) and (df2 <= df3) and (df3 != df4)')
np.allclose(result1, result3)
# True

result1 = (df1 < 0.5) & (df2 < 0.5) | (df3 < df4)
#         0      1      2
# 0    True   True  False
# 1   False   True  False
# 2   False   True  False
# 3    True   True   True
# 4   False  False  False
# ..    ...    ...    ...
# 95   True  False  False
# 96   True  False   True
# 97  False  False   True
# 98   True   True  False
# 99   True  False   True
result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
np.allclose(result1, result2)
# True
result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
np.allclose(result1, result3)
# True

# object attributes and indices
result1 = df2.T[0] + df3.iloc[1]
result2 = pd.eval('df2.T[0] + df3.iloc[1]')
np.allclose(result1, result2)
# True

# Other operations are not implemented in pd.eval currently.
# Use NumExpr in those cases.

# DataFrame.eval for column-wise operations

df = pd.DataFrame(rng.random((1000, 3)), columns=['A', 'B', 'C'])
df.head()
#           A         B         C
# 0  0.491509  0.026443  0.493859
# 1  0.013433  0.288272  0.457417
# 2  0.239635  0.482827  0.683364
# 3  0.312111  0.930706  0.614875
# 4  0.312779  0.148760  0.354428

# using pd.eval
result1 = (df['A'] + df['B']) / (df['C'] - 1)
result2 = pd.eval("(df.A + df.B) / (df.C - 1)")
np.allclose(result1, result2)
# True

# using Dataframe.eval:
# column names as variables within the evaluated expression
result3 = df.eval('(A + B) / (C - 1)')
np.allclose(result1, result3)
# True

# assignment in DataFrame.eval

df.head()
#           A         B         C
# 0  0.491509  0.026443  0.493859
# 1  0.013433  0.288272  0.457417
# 2  0.239635  0.482827  0.683364
# 3  0.312111  0.930706  0.614875
# 4  0.312779  0.148760  0.354428

# Add a new column
df.eval('D = (A + B) / C', inplace=True)
df.head()
#           A         B         C         D
# 0  0.491509  0.026443  0.493859  1.048786
# 1  0.013433  0.288272  0.457417  0.659583
# 2  0.239635  0.482827  0.683364  1.057214
# 3  0.312111  0.930706  0.614875  2.021251
# 4  0.312779  0.148760  0.354428  1.302209

# Modify the column
df.eval('D = (A - B) / C', inplace=True)
df.head()
#           A         B         C         D
# 0  0.491509  0.026443  0.493859  0.941698
# 1  0.013433  0.288272  0.457417 -0.600850
# 2  0.239635  0.482827  0.683364 -0.355876
# 3  0.312111  0.930706  0.614875 -1.006050
# 4  0.312779  0.148760  0.354428  0.462772

# Local variables in DataFrame.eval by
# the @ character that marks a variable name

column_mean = df.mean(1)
result1 = df['A'] + column_mean
result2 = df.eval('A + @column_mean')
np.allclose(result1, result2)
# True
