import numpy as np
import pandas as pd


from utils.common import make_df
from utils.display import display


make_df('ABC', range(3))
#     A   B   C
# 0  A0  B0  C0
# 1  A1  B1  C1
# 2  A2  B2  C2

# Recall: concatenation in NumPy
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])
# array([1, 2, 3, 4, 5, 6, 7, 8, 9])

x = [[1, 2],
     [3, 4]]
# Concatenate as additional rows:
np.concatenate([x, x])
# array([[1, 2],
#        [3, 4],
#        [1, 2],
#        [3, 4]])
# Concatenate as additional columns:
np.concatenate([x, x], axis=1)
# array([[1, 2, 1, 2],
#        [3, 4, 3, 4]])

# pd.concat
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
# 1    A
# 2    B
# 3    C
# dtype: object
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
# 4    D
# 5    E
# 6    F
# dtype: object

pd.concat([ser1, ser2])
# 1    A
# 2    B
# 3    C
# 4    D
# 5    E
# 6    F
# dtype: object

pd.concat([ser1, ser2], axis=1)
#      0    1
# 1    A  NaN
# 2    B  NaN
# 3    C  NaN
# 4  NaN    D
# 5  NaN    E
# 6  NaN    F

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
display('df1', 'df2', 'pd.concat([df1, df2])')
# df1
#     A   B
# 1  A1  B1
# 2  A2  B2
#
# df2
#     A   B
# 3  A3  B3
# 4  A4  B4
#
# pd.concat([df1, df2])
#     A   B
# 1  A1  B1
# 2  A2  B2
# 3  A3  B3
# 4  A4  B4

df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
display('df3', 'df4', "pd.concat([df3, df4], axis='columns')")
# df3
#     A   B
# 0  A0  B0
# 1  A1  B1
#
# df4
#     C   D
# 0  C0  D0
# 1  C1  D1
#
# pd.concat([df3, df4], axis='columns')
#     A   B   C   D
# 0  A0  B0  C0  D0
# 1  A1  B1  C1  D1

display('df3', 'df4', "pd.concat([df3, df4])")
# df3
#     A   B
# 0  A0  B0
# 1  A1  B1
#
# df4
#     C   D
# 0  C0  D0
# 1  C1  D1
#
# pd.concat([df3, df4])
#      A    B    C    D
# 0   A0   B0  NaN  NaN
# 1   A1   B1  NaN  NaN
# 0  NaN  NaN   C0   D0
# 1  NaN  NaN   C1   D1

x = make_df('AB', [0, 1])
y = make_df('AB', [0, 1])

# Duplicate the index:
display('x', 'y', 'pd.concat([x, y])')
# x
#     A   B
# 0  A0  B0
# 1  A1  B1
#
# y
#     A   B
# 0  A0  B0
# 1  A1  B1
#
# pd.concat([x, y])
#     A   B
# 0  A0  B0
# 1  A1  B1
# 0  A0  B0
# 1  A1  B1

# Treat repeated indices as an error:
try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)
# ValueError: Indexes have overlapping values: Index([0, 1], dtype='int64')

# Ignore the index
display('x', 'y', 'pd.concat([x, y], ignore_index=True)')
# x
#     A   B
# 0  A0  B0
# 1  A1  B1
#
# y
#     A   B
# 0  A0  B0
# 1  A1  B1
#
# pd.concat([x, y], ignore_index=True)
#     A   B
# 0  A0  B0
# 1  A1  B1
# 2  A0  B0
# 3  A1  B1

# Add MultiIndex keys
display('x', 'y', "pd.concat([x, y], keys=['x', 'y'])")
# x
#     A   B
# 0  A0  B0
# 1  A1  B1
#
# y
#     A   B
# 0  A0  B0
# 1  A1  B1
#
# pd.concat([x, y], keys=['x', 'y'])
#       A   B
# x 0  A0  B0
#   1  A1  B1
# y 0  A0  B0
#   1  A1  B1

# concatenation with joins
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
display('df5', 'df6', 'pd.concat([df5, df6])')
# df5
#     A   B   C
# 1  A1  B1  C1
# 2  A2  B2  C2
#
# df6
#     B   C   D
# 3  B3  C3  D3
# 4  B4  C4  D4
#
# pd.concat([df5, df6])
#      A   B   C    D
# 1   A1  B1  C1  NaN
# 2   A2  B2  C2  NaN
# 3  NaN  B3  C3   D3
# 4  NaN  B4  C4   D4

display('df5', 'df6', "pd.concat([df5, df6], join='inner')")
# df5
#     A   B   C
# 1  A1  B1  C1
# 2  A2  B2  C2
#
# df6
#     B   C   D
# 3  B3  C3  D3
# 4  B4  C4  D4
#
# pd.concat([df5, df6], join='inner')
#     B   C
# 1  B1  C1
# 2  B2  C2
# 3  B3  C3
# 4  B4  C4

df6.reindex(df5.columns, axis=1)
#     A   B   C
# 3 NaN  B3  C3
# 4 NaN  B4  C4

pd.concat([df5, df6.reindex(df5.columns, axis=1)])
#      A   B   C
# 1   A1  B1  C1
# 2   A2  B2  C2
# 3  NaN  B3  C3
# 4  NaN  B4  C4
