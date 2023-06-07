import numpy as np
import pandas as pd

pd.Series([1, np.nan, 2, None])
# 0    1.0
# 1    NaN
# 2    2.0
# 3    NaN
# dtype: float64

x = pd.Series(range(2), dtype=int)
# 0    0
# 1    1
# dtype: int64

x[0] = None
x
# 0    NaN
# 1    1.0
# dtype: float64

# nullable dtypes, e.g., pd.Int32

pd.Series([1, np.nan, 2, None, pd.NA], dtype='Int32')
# 0       1
# 1    <NA>
# 2       2
# 3    <NA>
# 4    <NA>
# dtype: Int32

# Pandas treats None, NaN, and NA as essentially
# interchangeable for indicating missing or null values.

# isnull: generates a Boolean mask indicating missing values
# notnull: opposite of isnull
# dropna: returns a filtered version of the data
# fillna: returns a copy of the data with missing values filled or imputed

data = pd.Series([1, np.nan, 'hello', None])
# 0        1
# 1      NaN
# 2    hello
# 3     None
# dtype: object

data[data.notnull()]
# 0        1
# 2    hello
# dtype: object

data.dropna()
# 0        1
# 2    hello
# dtype: object

df = pd.DataFrame(
    [[1, np.nan, 2],
     [2, 3, 5],
     [np.nan, 4, 6]]
)
#      0    1  2
# 0  1.0  NaN  2
# 1  2.0  3.0  5
# 2  NaN  4.0  6

# These lines are equivalent:
df.dropna()
df.dropna(axis=0)
df.dropna(axis='rows')
df.dropna(axis='rows', how='any')
#      0    1  2
# 1  2.0  3.0  5

# These lines are equivalent:
df.dropna(axis=1)
df.dropna(axis='columns')
df.dropna(axis='columns', how='any')
#    2
# 0  2
# 1  5
# 2  6

df.dropna(axis='columns', how='all')
#      0    1  2
# 0  1.0  NaN  2
# 1  2.0  3.0  5
# 2  NaN  4.0  6

df[3] = np.nan
df
#      0    1  2   3
# 0  1.0  NaN  2 NaN
# 1  2.0  3.0  5 NaN
# 2  NaN  4.0  6 NaN

df.dropna(axis='columns', how='all')
#      0    1  2
# 0  1.0  NaN  2
# 1  2.0  3.0  5
# 2  NaN  4.0  6

# The thresh parameter specifies a minimum number
# of non-null values for the row/column to be kept:

df.dropna(axis='rows', thresh=3)
#      0    1  2   3
# 1  2.0  3.0  5 NaN

df.dropna(axis='rows', thresh=2)
#      0    1  2   3
# 0  1.0  NaN  2 NaN
# 1  2.0  3.0  5 NaN
# 2  NaN  4.0  6 NaN

df.dropna(axis='rows', thresh=4)
# Empty DataFrame
# Columns: [0, 1, 2, 3]
# Index: []

df.loc[2, 0] = np.nan
#      0    1    2   3
# 0  1.0  NaN  NaN NaN
# 1  2.0  3.0  5.0 NaN
# 2  NaN  4.0  6.0 NaN

df.dropna(axis='rows', thresh=2)
#      0    1    2   3
# 1  2.0  3.0  5.0 NaN
# 2  NaN  4.0  6.0 NaN

data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'), dtype='Int32')
# a       1
# b    <NA>
# c       2
# d    <NA>
# e       3
# dtype: Int32

# forward fill
data.fillna(method='ffill')
# a    1
# b    1
# c    2
# d    2
# e    3
# dtype: Int32

# back fill
data.fillna(method='bfill')
# a    1
# b    2
# c    2
# d    3
# e    3
# dtype: Int32

data.fillna(0)
# a    1
# b    0
# c    2
# d    0
# e    3
# dtype: Int32

data = pd.Series([np.nan, 1, 2, 3, None], index=list('abcde'), dtype='Int32')
# a    <NA>
# b       1
# c       2
# d       3
# e    <NA>
# dtype: Int32

data.fillna(method='ffill')
# a    <NA>
# b       1
# c       2
# d       3
# e       3
# dtype: Int32

data.fillna(method='bfill')
# a       1
# b       1
# c       2
# d       3
# e    <NA>
# dtype: Int32

df = pd.DataFrame(
    [[1, np.nan, np.nan],
     [2, 3, np.nan],
     [np.nan, 4, np.nan]]
)
#      0    1   2
# 0  1.0  NaN NaN
# 1  2.0  3.0 NaN
# 2  NaN  4.0 NaN

# equivalent
df.fillna(method='ffill', axis=1)
df.fillna(method='ffill', axis='columns')  # each column in a row
#      0    1    2
# 0  1.0  1.0  1.0
# 1  2.0  3.0  3.0
# 2  NaN  4.0  4.0

# equivalent
df.fillna(method='ffill', axis=0)
df.fillna(method='ffill', axis='rows')  # each row in a column
#      0    1   2
# 0  1.0  NaN NaN
# 1  2.0  3.0 NaN
# 2  2.0  4.0 NaN

