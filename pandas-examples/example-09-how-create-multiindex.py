import numpy as np
import pandas as pd

df = pd.DataFrame(
    np.random.rand(4, 2),
    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
    columns=['data1', 'data2']
)
#         data1     data2
# a 1  0.568097  0.683264
#   2  0.093471  0.150092
# b 1  0.144490  0.165862
#   2  0.672473  0.439458

data = {
    ('California', 2010): 37253956,
    ('California', 2020): 39538223,
    ('New York', 2010): 19378102,
    ('New York', 2020): 20201249,
    ('Texas', 2010): 25145561,
    ('Texas', 2020): 29145505,
}

pd.Series(data)
# California  2010    37253956
#             2020    39538223
# New York    2010    19378102
#             2020    20201249
# Texas       2010    25145561
#             2020    29145505
# dtype: int64

# These lines of code are equivalent:
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
pd.MultiIndex(levels=[['a', 'b'], [1, 2]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
# MultiIndex([('a', 1),
#             ('a', 2),
#             ('b', 1),
#             ('b', 2)],
#            )

# pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b', 'c'], [1, 2, 1, 2]])
# ValueError: all arrays must be same length

data = ["aaa", "bbb", "ccc", "ddd"]
ser = pd.Series(data, index=pd.MultiIndex.from_product([['a', 'b'], [1, 2]]))
# a  1    aaa
#    2    bbb
# b  1    ccc
#    2    ddd
# dtype: object

ser.index.names = ['char', 'num']
ser
# char  num
# a     1      aaa
#       2      bbb
# b     1      ccc
#       2      ddd
# dtype: object

# hierarchical indices and columns
index = pd.MultiIndex.from_product(
    [[2013, 2014], [1, 2]],
    names=['year', 'visit']
)
# MultiIndex([(2013, 1),
#             (2013, 2),
#             (2014, 1),
#             (2014, 2)],
#            names=['year', 'visit'])
columns = pd.MultiIndex.from_product(
    [['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
    names=['subject', 'type']
)
# MultiIndex([(  'Bob',   'HR'),
#             (  'Bob', 'Temp'),
#             ('Guido',   'HR'),
#             ('Guido', 'Temp'),
#             (  'Sue',   'HR'),
#             (  'Sue', 'Temp')],
#            names=['subject', 'type'])

# Mock some data
data = np.round(np.random.randn(4, 6), 1)
# array([[ 1.3,  0.3, -1.5, -0.6,  0.3, -1.1],
#        [ 0.7, -1.9, -0.2, -0. , -0.2, -0.5],
#        [ 1.8, -0.1,  0.5,  1.8, -1.3,  1.1],
#        [-1.2, -1.5,  1.4, -1.1,  1.6,  0.3]])

data[:, ::2] *= 10
# array([[ 13. ,   0.3, -15. ,  -0.6,   3. ,  -1.1],
#        [  7. ,  -1.9,  -2. ,  -0. ,  -2. ,  -0.5],
#        [ 18. ,  -0.1,   5. ,   1.8, -13. ,   1.1],
#        [-12. ,  -1.5,  14. ,  -1.1,  16. ,   0.3]])

data += 37
# array([[50. , 37.3, 22. , 36.4, 40. , 35.9],
#        [44. , 35.1, 35. , 37. , 35. , 36.5],
#        [55. , 36.9, 42. , 38.8, 24. , 38.1],
#        [25. , 35.5, 51. , 35.9, 53. , 37.3]])

# Create the DataFrame (four-dimensional data)
health_data = pd.DataFrame(data, index=index, columns=columns)
# subject      Bob       Guido         Sue
# type          HR  Temp    HR  Temp    HR  Temp
# year visit
# 2013 1      50.0  37.3  22.0  36.4  40.0  35.9
#      2      44.0  35.1  35.0  37.0  35.0  36.5
# 2014 1      55.0  36.9  42.0  38.8  24.0  38.1
#      2      25.0  35.5  51.0  35.9  53.0  37.3

health_data['Guido']
# type          HR  Temp
# year visit
# 2013 1      22.0  36.4
#      2      35.0  37.0
# 2014 1      42.0  38.8
#      2      51.0  35.9
