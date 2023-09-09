import pandas as pd

ind = pd.Index([2, 3, 5, 7, 11])
# Index([2, 3, 5, 7, 11], dtype='int64')

ind[1]
# 3

ind[::2]
# Index([2, 5, 11], dtype='int64')

print(ind.size, ind.shape, ind.ndim, ind.dtype)
# 5 (5,) 1 int64

# immutable objects
# ind[1] = 99
# TypeError: Index does not support mutable operations

ind_a = pd.Index([1, 3, 5, 7, 9])
# Index([1, 3, 5, 7, 9], dtype='int64')
ind_b = pd.Index([2, 3, 5, 7, 11])
# Index([2, 3, 5, 7, 11], dtype='int64')
ind_a.intersection(ind_b)
# Index([3, 5, 7], dtype='int64')
ind_a.union(ind_b)
# Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')
ind_a.difference(ind_b)
# Index([1, 9], dtype='int64')
ind_a.symmetric_difference(ind_b)
# Index([1, 2, 9, 11], dtype='int64')
