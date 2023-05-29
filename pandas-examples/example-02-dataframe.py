import numpy as np
import pandas as pd

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

area_dict = {
    'California': 423967,
    'Texas': 695662,
    'Florida': 170312,
    'New York': 141297,
    'Pennsylvania': 119280
}
area = pd.Series(area_dict)
# California      423967
# Texas           695662
# Florida         170312
# New York        141297
# Pennsylvania    119280
# dtype: int64

states = pd.DataFrame({'population': population, 'area': area})
#               population    area
# California      39538223  423967
# Texas           29145505  695662
# Florida         21538187  170312
# New York        20201249  141297
# Pennsylvania    13002700  119280

states.index
# Index(['California', 'Texas', 'Florida', 'New York', 'Pennsylvania'], dtype='object')

states.columns
# Index(['population', 'area'], dtype='object')

# Returns the Series object:
states['area']
# California      423967
# Texas           695662
# Florida         170312
# New York        141297
# Pennsylvania    119280
# Name: area, dtype: int64

# states[0]
# KeyError: 0

# Create from a single Series object:
pd.DataFrame(population, columns=['population'])
#               population
# California      39538223
# Texas           29145505
# Florida         21538187
# New York        20201249
# Pennsylvania    13002700

# Create from a list of dictionaries:
data = [{'a': i, 'b': 2 * i} for i in range(3)]
# [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'b': 4}]
pd.DataFrame(data)
#    a  b
# 0  0  0
# 1  1  2
# 2  2  4

# Create from a list of dictionaries with missing values:
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
#      a  b    c
# 0  1.0  2  NaN
# 1  NaN  3  4.0

# Create from a dictionary of Series objects:
pd.DataFrame({'population': population, 'area': area})
#               population    area
# California      39538223  423967
# Texas           29145505  695662
# Florida         21538187  170312
# New York        20201249  141297
# Pennsylvania    13002700  119280

# Create from a two-dimensional NumPy array:
pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c'])
#         foo       bar
# a  0.027877  0.467922
# b  0.059756  0.164886
# c  0.255604  0.774385

# Create from a NumPy structured array:
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
# array([(0, 0.), (0, 0.), (0, 0.)], dtype=[('A', '<i8'), ('B', '<f8')])
pd.DataFrame(A)
#    A    B
# 0  0  0.0
# 1  0  0.0
# 2  0  0.0

pd.DataFrame(A, index=["a", "b", "c"])
#    A    B
# a  0  0.0
# b  0  0.0
# c  0  0.0
