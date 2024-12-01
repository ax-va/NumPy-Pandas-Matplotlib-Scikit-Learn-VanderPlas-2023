import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
ser = pd.Series(rng.integers(0, 10, 4))
# 0    0
# 1    7
# 2    6
# 3    4
# dtype: int64

df = pd.DataFrame(rng.integers(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])
#    A  B  C  D
# 0  4  8  0  6
# 1  2  0  5  9
# 2  7  7  7  7

# NumPy ufunc returns another Pandas object preserving indices:

np.exp(ser)
# 0       1.000000
# 1    1096.633158
# 2     403.428793
# 3      54.598150
# dtype: float64

np.exp(df)
#              A            B            C            D
# 0    54.598150  2980.957987     1.000000   403.428793
# 1     7.389056     1.000000   148.413159  8103.083928
# 2  1096.633158  1096.633158  1096.633158  1096.633158

np.sin(df * np.pi / 4)
#               A             B         C         D
# 0  1.224647e-16 -2.449294e-16  0.000000 -1.000000
# 1  1.000000e+00  0.000000e+00 -0.707107  0.707107
# 2 -7.071068e-01 -7.071068e-01 -0.707107 -0.707107

area = pd.Series(
    {
        'Alaska': 1723337,
        'Texas': 695662,
        'California': 423967
    },
    name='area'
)
# Alaska        1723337
# Texas          695662
# California     423967
# Name: area, dtype: int64

population = pd.Series(
    {
        'California': 39538223,
        'Texas': 29145505,
        'Florida': 21538187
    },
    name='population'
)
# California    39538223
# Texas         29145505
# Florida       21538187
# Name: population, dtype: int64

# The resulting Series contains the union of indices of the two Series:

population / area
# Alaska              NaN
# California    93.257784
# Florida             NaN
# Texas         41.896072
# dtype: float64

area.index.union(population.index)
# Index(['Alaska', 'California', 'Florida', 'Texas'], dtype='object')

# NaN for missing data:

A = pd.Series([2, 4, 6], index=[0, 1, 2])
# 0    2
# 1    4
# 2    6
# dtype: int64

B = pd.Series([1, 3, 5], index=[1, 2, 3])
# 1    1
# 2    3
# 3    5
# dtype: int64

A + B
# 0    NaN
# 1    5.0
# 2    9.0
# 3    NaN
# dtype: float64

# If a value is missing, it is replaced by 0:

A.add(B, fill_value=0)
# 0    2.0
# 1    5.0
# 2    9.0
# 3    5.0
# dtype: float64

A = pd.DataFrame(rng.integers(0, 20, (2, 2)), columns=['a', 'b'])
#     a  b
# 0  10  2
# 1  16  9

B = pd.DataFrame(rng.integers(0, 10, (3, 3)), columns=['b', 'a', 'c'])
#    b  a  c
# 0  5  3  1
# 1  9  7  6
# 2  4  8  5

A + B
#       a     b   c
# 0  13.0   7.0 NaN
# 1  23.0  18.0 NaN
# 2   NaN   NaN NaN

A.values.mean()
# 9.25

A.add(B, fill_value=A.values.mean())
#        a      b      c
# 0  13.00   7.00  10.25
# 1  23.00  18.00  15.25
# 2  17.25  13.25  14.25

# Python operator       Pandas method(s)
# +                     add
# -                     sub, subtract
# *                     mul, multiply
# /                     truediv, div, divide
# //                    floordiv
# %                     mod
# **                    pow

# NumPy broadcasting applied row-wise:

A = rng.integers(10, size=(3, 4))
# array([[4, 4, 2, 0],
#        [5, 8, 0, 8],
#        [8, 2, 6, 1]])

A[0]
# array([4, 4, 2, 0])

A + A[0]
# array([[ 8,  8,  4,  0],
#        [ 9, 12,  2,  8],
#        [12,  6,  8,  1]])

# Pandas broadcasting applied row-wise:

df = pd.DataFrame(A, columns=['Q', 'R', 'S', 'T'])
#    Q  R  S  T
# 0  4  4  2  0
# 1  5  8  0  8
# 2  8  2  6  1

df.iloc[0]
# Q    4
# R    4
# S    2
# T    0
# Name: 0, dtype: int64

df - df.iloc[0]
#    Q  R  S  T
# 0  0  0  0  0
# 1  1  4 -2  8
# 2  4 -2  4  1

# column-wise:

df.subtract(df['R'], axis=0)
#    Q  R  S  T
# 0  0  0 -2 -4
# 1 -3  0 -8  0
# 2  6  0  4 -1

# 'axis' is used to compare by index or columns.
# default: axis='columns'
# axis: 0 or 'index', 1 or 'columns'

halfrow = df.iloc[0, ::2]
# Q    4
# S    2
# Name: 0, dtype: int64

df - halfrow
#      Q   R    S   T
# 0  0.0 NaN  0.0 NaN
# 1  1.0 NaN -2.0 NaN
# 2  4.0 NaN  4.0 NaN
