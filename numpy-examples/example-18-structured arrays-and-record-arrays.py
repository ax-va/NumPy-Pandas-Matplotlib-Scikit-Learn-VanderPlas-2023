import numpy as np

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={"names": ("name", "age", "weight"), "formats": ("U10", "i4", "f8")})
# U10: Unicode string of maximum length 10
# i4: 4-byte (32-bit) integer
# f8: 8-byte (64-bit) float
print(data.dtype)  # [('name', '<U10'), ('age', '<i4'), ('weight', '<f8')]
# <: little endian
# >: big endian

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
data['name'] = name
data['age'] = age
data['weight'] = weight

data
# array([('Alice', 25, 55. ), ('Bob', 45, 85.5), ('Cathy', 37, 68. ),
#        ('Doug', 19, 61.5)],
#       dtype=[('name', '<U10'), ('age', '<i4'), ('weight', '<f8')])

# Get all names
data['name']
# array(['Alice', 'Bob', 'Cathy', 'Doug'], dtype='<U10')
# Get first row of data
data[0]
# ('Alice', 25, 55.)

# Get the name from the last row
data[-1]['name']
# 'Doug'

# Get names where age is under 30
data[data['age'] < 30]['name']
# array(['Alice', 'Doug'], dtype='<U10')

# examples of dtypes

np.dtype({'names': ('name', 'age', 'weight'), 'formats': ((np.str_, 10), int, np.float32)})
# dtype([('name', '<U10'), ('age', '<i8'), ('weight', '<f4')])

np.dtype({'names': ('name', 'age', 'weight'), 'formats': (str, np.int32, float)})
# dtype([('name', '<U'), ('age', '<i4'), ('weight', '<f8')])

np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
# dtype([('name', 'S10'), ('age', '<i4'), ('weight', '<f8')])

np.dtype('S10, i4, f8')
# dtype([('f0', 'S10'), ('f1', '<i4'), ('f2', '<f8')])

# more advanced compound dtypes

tp = np.dtype([('ID', 'i8'), ('mat', 'f8', (3, 3))])
# dtype([('ID', '<i8'), ('mat', '<f8', (3, 3))])
X = np.zeros(1, dtype=tp)
# array([(0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])],
#       dtype=[('ID', '<i8'), ('mat', '<f8', (3, 3))])
X[0]
# (0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
X['mat'][0]
# array([[0., 0., 0.],
#        [0., 0., 0.],
#        [0., 0., 0.]])

# Fields can be accessed as attributes rather than as dictionary keys:
data_rec = data.view(np.recarray)
# rec.array([('Alice', 25, 55. ), ('Bob', 45, 85.5), ('Cathy', 37, 68. ),
#            ('Doug', 19, 61.5)],
#           dtype=[('name', '<U10'), ('age', '<i4'), ('weight', '<f8')])
data_rec.age
# array([25, 45, 37, 19], dtype=int32)

# %timeit data['age']
# 76.3 ns ± 0.408 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
# %timeit data_rec['age']
# 1.16 µs ± 8.41 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
# %timeit data_rec.age
# 2.04 µs ± 1.64 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
