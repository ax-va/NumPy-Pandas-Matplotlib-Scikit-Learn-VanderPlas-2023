import numpy as np

rng = np.random.default_rng(seed=42)
big_array = rng.random(1_000_000)
small_array = rng.random(1_000)

# sum
print(np.sum(big_array))  # 500026.47617408895
# In IPython:
# %timeit sum(big_array)
# 103 ms ± 21.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# %timeit np.sum(big_array)
# 2.44 ms ± 53.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# %timeit sum(small_array)
# 112 µs ± 27.5 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
# %timeit np.sum(small_array)
# 7.12 µs ± 657 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

# min, max
print(np.min(big_array), np.max(big_array))
# 1.2500323287589765e-07 0.9999997172035572
# In IPython:
# %timeit min(big_array)
# 98.7 ms ± 7.24 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# %timeit max(big_array)
# 98.1 ms ± 12.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# shorthands:
print(big_array.max(), big_array.min(), big_array.sum())
# 0.9999997172035572 1.2500323287589765e-07 500026.47617408895

# Create a matrix:
# array([[4, 1, 6, 7],
#        [2, 7, 4, 3],
#        [6, 2, 3, 7]], dtype=int64)
matrix = rng.integers(0, 10, (3, 4))

# Sum out all entries
print(matrix.sum())  # 52
# Sum out all rows:
# array([12, 10, 13, 17], dtype=int64)
matrix.sum(axis=0)
# Sum out all columns:
# array([18, 16, 18], dtype=int64)
matrix.sum(axis=1)

# min
print(matrix.min())  # 1
# min of rows:
# array([2, 1, 3, 3], dtype=int64)
matrix.min(axis=0)  #
# max
print(matrix.max())  # 7

