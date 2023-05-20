import numpy as np

# Fields can be accessed as attributes rather than as dictionary keys

data = np.zeros(4, dtype={"names": ("name", "age", "weight"), "formats": ("U10", "i4", "f8")})
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
data['name'] = name
data['age'] = age
data['weight'] = weight
data['age']
# array([25, 45, 37, 19], dtype=int32)

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
