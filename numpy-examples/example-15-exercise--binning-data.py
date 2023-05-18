import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)
x = rng.normal(size=100)

# Compute a histogram by hand
bins = np.linspace(-5, 5, 20)
# array([-5.        , -4.47368421, -3.94736842, -3.42105263, -2.89473684,
#        -2.36842105, -1.84210526, -1.31578947, -0.78947368, -0.26315789,
#         0.26315789,  0.78947368,  1.31578947,  1.84210526,  2.36842105,
#         2.89473684,  3.42105263,  3.94736842,  4.47368421,  5.        ])
counts = np.zeros_like(bins)
# array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0.])

# Find the appropriate bin for each x
i = np.searchsorted(bins, x)

# Add 1 to each of these bins
np.add.at(counts, i, 1)

# Plot the results
plt.plot(bins, counts, drawstyle='steps')
plt.savefig('../figures/binning-data-1.svg')
plt.close()

# But there is another method.
# Use the hist method.
# It uses np.histogram.
plt.hist(x, bins, histtype='step')
plt.savefig('../figures/binning-data-2.svg')
plt.close()

# %timeit counts, edges = np.histogram(x, bins)
# 14.8 µs ± 97.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
# %timeit np.add.at(counts, np.searchsorted(bins, x), 1)
# 11.9 µs ± 32.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

x = rng.normal(size=1_000_000)
# %timeit counts, edges = np.histogram(x, bins)
# 58 ms ± 25.6 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# %timeit np.add.at(counts, np.searchsorted(bins, x), 1)
# Problem
