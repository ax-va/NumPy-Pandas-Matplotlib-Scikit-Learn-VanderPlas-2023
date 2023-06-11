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
ser.index.names = ['state', 'year']
ser
# state       year
# California  2010    37253956
#             2020    39538223
# New York    2010    19378102
#             2020    20201249
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

ser.index
# MultiIndex([('California', 2010),
#             ('California', 2020),
#             (  'New York', 2010),
#             (  'New York', 2020),
#             (     'Texas', 2010),
#             (     'Texas', 2020)],
#            names=['state', 'year'])

# equivalent
ser.unstack()
ser.unstack(level=1)  # The second multiindex is transformed to columns
# year            2010      2020
# state
# California  37253956  39538223
# New York    19378102  20201249
# Texas       25145561  29145505

ser.unstack().index
# Index(['California', 'New York', 'Texas'], dtype='object', name='state')
ser.unstack().columns
# Index([2010, 2020], dtype='int64', name='year')

ser.unstack(level=0)  # The first multiindex is transformed to columns
# state  California  New York     Texas
# year
# 2010     37253956  19378102  25145561
# 2020     39538223  20201249  29145505

ser.unstack(level=0).index
# Index([2010, 2020], dtype='int64', name='year')

ser.unstack(level=0).columns
# Index(['California', 'New York', 'Texas'], dtype='object', name='state')

ser.unstack().stack()
# state       year
# California  2010    37253956
#             2020    39538223
# New York    2010    19378102
#             2020    20201249
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

ser.unstack().stack().index
# MultiIndex([('California', 2010),
#             ('California', 2020),
#             (  'New York', 2010),
#             (  'New York', 2020),
#             (     'Texas', 2010),
#             (     'Texas', 2020)],
#            names=['state', 'year'])

ser.reset_index()
#         state  year         0
# 0  California  2010  37253956
# 1  California  2020  39538223
# 2    New York  2010  19378102
# 3    New York  2020  20201249
# 4       Texas  2010  25145561
# 5       Texas  2020  29145505

pop_flat = ser.reset_index(name='population')
#         state  year  population
# 0  California  2010    37253956
# 1  California  2020    39538223
# 2    New York  2010    19378102
# 3    New York  2020    20201249
# 4       Texas  2010    25145561
# 5       Texas  2020    29145505

# Return a new multiindexed DataFrame
pop_flat.set_index(['state', 'year'])  # pop_flat remains unchanged
#                  population
# state      year
# California 2010    37253956
#            2020    39538223
# New York   2010    19378102
#            2020    20201249
# Texas      2010    25145561
#            2020    29145505
