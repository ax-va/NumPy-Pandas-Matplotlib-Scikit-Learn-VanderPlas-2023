import numpy as np
import pandas as pd

index = [
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

# Bad way:

pop = pd.Series(populations, index=index)
# (California, 2010)    37253956
# (California, 2020)    39538223
# (New York, 2010)      19378102
# (New York, 2020)      20201249
# (Texas, 2010)         25145561
# (Texas, 2020)         29145505
# dtype: int64

pop[('California', 2020):('Texas', 2010)]
# (California, 2020)    39538223
# (New York, 2010)      19378102
# (New York, 2020)      20201249
# (Texas, 2010)         25145561
# dtype: int64

pop[[i for i in pop.index if i[1] == 2010]]
# (California, 2010)    37253956
# (New York, 2010)      19378102
# (Texas, 2010)         25145561
# dtype: int64

# Better way:

index = pd.MultiIndex.from_tuples(index)
# MultiIndex([('California', 2010),
#             ('California', 2020),
#             (  'New York', 2010),
#             (  'New York', 2020),
#             (     'Texas', 2010),
#             (     'Texas', 2020)],
#            )

pop = pop.reindex(index)
# California  2010    37253956
#             2020    39538223
# New York    2010    19378102
#             2020    20201249
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

pop[:, 2020]
# California    39538223
# New York      20201249
# Texas         29145505
# dtype: int64

pop_df = pop.unstack()
#                 2010      2020
# California  37253956  39538223
# New York    19378102  20201249
# Texas       25145561  29145505

pop_df.stack()
# California  2010    37253956
#             2020    39538223
# New York    2010    19378102
#             2020    20201249
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

pop_df = pd.DataFrame(
    {
        'total': pop,
        'under18': [9284094, 8898092, 4318033, 4181528, 6879014, 7432474]
    }
)
#                     total  under18
# California 2010  37253956  9284094
#            2020  39538223  8898092
# New York   2010  19378102  4318033
#            2020  20201249  4181528
# Texas      2010  25145561  6879014
#            2020  29145505  7432474

