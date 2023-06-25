import numpy as np
import pandas as pd


rng = np.random.RandomState(0)
df = pd.DataFrame(
    {
        'my_key': ['A', 'B', 'C', 'A', 'B', 'C'],
        'my_data1': range(6),
        'my_data2': rng.randint(0, 10, 6)
    },
    # columns=['my_data1', 'my_data2']  # Add only these columns
)
#   my_key  my_data1  my_data2
# 0      A         0         5
# 1      B         1         0
# 2      C         2         3
# 3      A         3         3
# 4      B         4         7
# 5      C         5         9

# aggregation

df.groupby('my_key').aggregate(['min', np.median, max])
#        my_data1            my_data2
#             min median max      min median max
# my_key
# A             0    1.5   3        3    4.0   5
# B             1    2.5   4        0    3.5   7
# C             2    3.5   5        3    6.0   9

df.groupby('my_key').aggregate(
    {
        'my_data1': 'min',
        'my_data2': 'max'
    }
)
#         my_data1  my_data2
# my_key
# A              0         5
# B              1         7
# C              2         9

# filtering

df['my_data2'].mean()
# 4.5

def filter_by_std(x):
    return x['my_data2'].std() > x['my_data2'].mean()

df.groupby('my_key').std()
#         my_data1  my_data2
# my_key
# A        2.12132  1.414214
# B        2.12132  4.949747
# C        2.12132  4.242641

df.groupby('my_key').filter(filter_by_std)
#   my_key  my_data1  my_data2
# 1      B         1         0
# 4      B         4         7

# transformation

def center_data(x):
    return x - x.mean()

df.groupby('my_key').transform(center_data)
#    my_data1  my_data2
# 0      -1.5       1.0
# 1      -1.5      -3.5
# 2      -1.5      -3.0
# 3       1.5      -1.0
# 4       1.5       3.5
# 5       1.5       3.0

# the apply method

def norm_by_my_data2(x):
    """
    Args:
        DataFrame of group values
    Returns:
        Pandas object or scalar
    """
    x['my_data1'] /= x['my_data2'].sum()
    return x

# equivalent
df.groupby('my_key').apply(norm_by_my_data2)
df.groupby('my_key', group_keys=True).apply(norm_by_my_data2)
#          my_key  my_data1  my_data2
# my_key
# A      0      A  0.000000         5
#        3      A  0.375000         3
# B      1      B  0.142857         0
#        4      B  0.571429         7
# C      2      C  0.166667         3
#        5      C  0.416667         9

df.groupby('my_key', group_keys=False).apply(norm_by_my_data2)
#   my_key  my_data1  my_data2
# 0      A  0.000000         5
# 1      B  0.142857         0
# 2      C  0.166667         3
# 3      A  0.375000         3
# 4      B  0.571429         7
# 5      C  0.416667         9

df2 = pd.DataFrame(
    {
        'A': 'a a b'.split(),
        'B': [1, 2, 3],
        'C': [4, 6, 5]
    }
)
#    A  B  C
# 0  a  1  4
# 1  a  2  6
# 2  b  3  5

df2.sum()
# A    aab
# B      6
# C     15
# dtype: object

df2.groupby('A', group_keys=False).apply(lambda x: x / x.sum())
#           B    C
# 0  0.333333  0.4
# 1  0.666667  0.6
# 2  1.000000  1.0

df2.groupby('A', group_keys=True).apply(lambda x: x / x.sum())
#             B    C
# A
# a 0  0.333333  0.4
#   1  0.666667  0.6
# b 2  1.000000  1.0

df2.groupby('A', group_keys=True).apply(lambda x: x)
#      A  B  C
# A
# a 0  a  1  4
#   1  a  2  6
# b 2  b  3  5
