import pandas as pd

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
# a    0.25
# b    0.50
# c    0.75
# d    1.00
# dtype: float64

data['a']
#  0.25

'a' in data
# True

data.keys()
# Index(['a', 'b', 'c', 'd'], dtype='object')

data.values
# array([0.25, 0.5 , 0.75, 1.  ])

list(data.items())
# [('a', 0.25), ('b', 0.5), ('c', 0.75), ('d', 1.0)]

data['e'] = 1.25
data
# a    0.25
# b    0.50
# c    0.75
# d    1.00
# e    1.25
# dtype: float64

# slices, masking, and fancy indexing

# slicing by explicit index
data['a':'c']  # By explicit index, the final index is included
# a    0.25
# b    0.50
# c    0.75
# dtype: float64

# slicing by implicit integer index
data[0:2]  # By implicit index, the final index is excluded
# a    0.25
# b    0.50
# dtype: float64

# masking
data[(data >= 0.25) & (data <= 0.75)]
# a    0.25
# b    0.50
# c    0.75
# dtype: float64

# fancy indexing
data[['a', 'e']]
# a    0.25
# e    1.25
# dtype: float64

data[[0, -1]]
# a    0.25
# e    1.25
# dtype: float64

# indexers: loc and iloc

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
# 1    a
# 3    b
# 5    c
# dtype: object

# explicit index when indexing
data[1]
# 'a'

# implicit index when slicing
data[1:3]
# 3    b
# 5    c
# dtype: object

# always the explicit index
data.loc[1]
# 'a'
data.loc[1:3]
# 1    a
# 3    b
# dtype: object

# always the implicit Python-style index
data.iloc[1]
# 'b'
data.iloc[1:3]
# 3    b
# 5    c
# dtype: object
