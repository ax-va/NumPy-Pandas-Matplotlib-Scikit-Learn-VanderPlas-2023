import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

# Kernel density estimation (KDE) uses a mixture consisting of one Gaussian component per point.

# # # motivation


def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x


x = make_data(1000)
# normalized histogram
hist = plt.hist(x, bins=30, density=True)  # density=True makes a probability distribution
plt.savefig('../scikit-learn-examples-figures/kernel-density-estimation-1--motivation-1--histogram.svg')
plt.close()

# The total area under the histogram is equal to 1
density, bins, patches = hist
widths = bins[1:] - bins[:-1]
(density * widths).sum()
# 1.0

# different interpretation of the data using histograms
x = make_data(20)
bins = np.linspace(-5, 10, 10)
# array([-5.        , -3.33333333, -1.66666667,  0.        ,  1.66666667,
#         3.33333333,  5.        ,  6.66666667,  8.33333333, 10.        ])
bins + 0.6
# array([-4.4       , -2.73333333, -1.06666667,  0.6       ,  2.26666667,
#         3.93333333,  5.6       ,  7.26666667,  8.93333333, 10.6       ])

fig, ax = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 4),
    sharex=True,
    sharey=True,
    subplot_kw={'xlim':(-4, 9), 'ylim':(-0.02, 0.3)}
)
fig.subplots_adjust(wspace=0.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, density=True)
    ax[i].plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.savefig('../scikit-learn-examples-figures/kernel-density-estimation-2--motivation-2--problem-with-histograms.svg')
plt.close()
