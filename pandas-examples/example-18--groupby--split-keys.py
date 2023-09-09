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

# A list, array, series, or index provide the grouping keys

L = [0, 1, 0, 1, 2, 0]
df.groupby(L).sum()
#   my_key  my_data1  my_data2
# 0    ACC         7        17
# 1     BA         4         3
# 2      B         4         7

# the first (A) goes into the group of 0
# the second (B) goes into the group of 1
# the third (C) goes into the group of 0
# the forth (A) goes into the group of 1
# the fifth (B) goes into the group of 2
# the sixth (C) goes into the group of 0

df.groupby(df['my_key']).sum()
#         my_data1  my_data2
# my_key
# A              3         8
# B              5         7
# C              7        12

# A dictionary or series maps index to group

df2 = df.set_index('my_key')
#         my_data1  my_data2
# my_key
# A              0         5
# B              1         0
# C              2         3
# A              3         3
# B              4         7
# C              5         9

mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}

df2.groupby(mapping).sum()
#            my_data1  my_data2
# my_key
# consonant        12        19
# vowel             3         8

# Any Python function can be applied that will
# input the index value and output the group

df2.groupby(str.lower).sum()
#         my_data1  my_data2
# my_key
# a              3         8
# b              5         7
# c              7        12

# A list of valid keys can be applied to group on a multi-index

df2.groupby([str.lower, mapping]).sum()
#                   my_data1  my_data2
# my_key my_key
# a      vowel             3         8
# b      consonant         5         7
# c      consonant         7        12
