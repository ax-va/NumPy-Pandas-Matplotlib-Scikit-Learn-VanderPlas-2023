import numpy as np
import pandas as pd

print(pd.__version__)  # 2.0.1

# pd.Series(data, index=index)
# data: lists, dictionaries, ndarrays
# index: optional

data = pd.Series([0.25, 0.5, 0.75, 1.0])
# 0    0.25
# 1    0.50
# 2    0.75
# 3    1.00
# dtype: float64

# familiar NumPy array:
data.values
# array([0.25, 0.5 , 0.75, 1.  ])

data.index
# RangeIndex(start=0, stop=4, step=1)

# implicitly defined integer index:
data[0]
# 0.25

data[1:3]
# 1    0.50
# 2    0.75
# dtype: float64

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
# a    0.25
# b    0.50
# c    0.75
# d    1.00
# dtype: float64

data.index
# Index(['a', 'b', 'c', 'd'], dtype='object')

# explicitly defined index:
data['a']
# 0.25

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 1, 4, 3])
# 2    0.25
# 1    0.50
# 4    0.75
# 3    1.00
# dtype: float64

# explicitly defined index:
data[2]
# 0.25

population_dict = {
    'California': 39538223,
    'Texas': 29145505,
    'Florida': 21538187,
    'New York': 20201249,
    'Pennsylvania': 13002700,
}
population = pd.Series(population_dict)
# California      39538223
# Texas           29145505
# Florida         21538187
# New York        20201249
# Pennsylvania    13002700
# dtype: int64

# explicitly defined index:
population['California']
# 39538223

population['California':'Florida']
# California    39538223
# Texas         29145505
# Florida       21538187
# dtype: int64

pd.Series({1: 'a', 2: 'b', 3: 'c'}, index=[2, 1])
# 2    b
# 1    a
# dtype: object

pd.Series(np.array([True, True, False]))
# 0     True
# 1     True
# 2    False
# dtype: bool
