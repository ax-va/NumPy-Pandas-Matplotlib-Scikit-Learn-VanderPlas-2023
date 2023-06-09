import numpy as np
import pandas as pd

tuples = [
    ('California', 2010),
    ('California', 2020),
    ('New York', 2010),
    ('New York', 2020),
    ('Texas', 2010),
    ('Texas', 2020),
]

populations = [
    37253956,
    39538223,
    19378102,
    20201249,
    25145561,
    29145505,
]

ser = pd.Series(populations, index=pd.MultiIndex.from_tuples(tuples))
# California  2010    37253956
#             2020    39538223
# New York    2010    19378102
#             2020    20201249
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

ser.index.names = ['state', 'year']
# ser
# state       year
# California  2010    37253956
#             2020    39538223
# New York    2010    19378102
#             2020    20201249
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

# indexing
ser['California', 2010]
# 37253956

# partial indexing
ser['California']
# year
# 2010    37253956
# 2020    39538223
# dtype: int64

# slicing
ser.loc['California':'New York']
# state       year
# California  2010    37253956
#             2020    39538223
# New York    2010    19378102
#             2020    20201249
# dtype: int64

# partial indexing on lower levels
ser[:, 2010]
# state
# California    37253956
# New York      19378102
# Texas         25145561
# dtype: int64

# masking
ser[ser > 25_000_000]
# state       year
# California  2010    37253956
#             2020    39538223
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

# fancy indexing
ser[['California', 'Texas']]
# state       year
# California  2010    37253956
#             2020    39538223
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

index = pd.MultiIndex.from_product(
    [[2013, 2014], [1, 2]],
    names=['year', 'visit']
)
columns = pd.MultiIndex.from_product(
    [['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
    names=['subject', 'type']
)
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)
# subject      Bob       Guido         Sue
# type          HR  Temp    HR  Temp    HR  Temp
# year visit
# 2013 1      32.0  37.1  38.0  35.9  34.0  37.6
#      2      44.0  34.9  53.0  36.1  25.0  37.8
# 2014 1      30.0  35.7  36.0  34.6  36.0  37.8
#      2      34.0  37.1  46.0  37.0  33.0  37.3

# equivalent
health_data['Guido', 'HR']
health_data.loc[:, ('Guido', 'HR')]
# year  visit
# 2013  1        38.0
#       2        53.0
# 2014  1        36.0
#       2        46.0
# Name: (Guido, HR), dtype: float64

health_data.iloc[:2, :2]
# subject      Bob
# type          HR  Temp
# year visit
# 2013 1      32.0  37.1
#      2      44.0  34.9

health_data.iloc[:3, :3]
# subject      Bob       Guido
# type          HR  Temp    HR
# year visit
# 2013 1      32.0  37.1  38.0
#      2      44.0  34.9  53.0
# 2014 1      30.0  35.7  36.0

health_data.loc[:, ('Bob', 'HR')]
# year  visit
# 2013  1        32.0
#       2        44.0
# 2014  1        30.0
#       2        34.0
# Name: (Bob, HR), dtype: float64

# health_data.loc[(:, 1), (:, 'HR')]
# SyntaxError: invalid syntax

idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]
# subject      Bob Guido   Sue
# type          HR    HR    HR
# year visit
# 2013 1      32.0  38.0  34.0
# 2014 1      30.0  36.0  36.0
