import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

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

# Plot two histograms with 'bins' and 'bins + offset' for *the same data*
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
plt.savefig('../scikit-learn-examples-figures/kernel-density-estimation-2--motivation-2--two-histograms-for-same-data.svg')
plt.close()
# On the left, the histogram shows a bimodal distribution.
# On the right, a unimodal distribution with a long tail.

# histogram as stack of blocks
fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle(xy=(edge, i), width=1, height=1, ec='black', alpha=0.5))
        ax.set_xlim(-4, 8)
        ax.set_ylim(-0.2, 8)
plt.savefig('../scikit-learn-examples-figures/kernel-density-estimation-3--motivation-3--histogram-as-stack-of-blocks.svg')
plt.close()

# Plot a “histogram” where blocks center on each individual point.
# This is an example of a kernel density estimate.
x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)
plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
plt.axis([-4, 8, -0.2, 8])
plt.savefig('../scikit-learn-examples-figures/kernel-density-estimation-4--motivation-4--density-histogram.svg')
plt.close()

# Smooth out the blocks.
# Use a standard normal curve at each point instead of a block.
# Plot a (non-normalized) kernel density estimate with a Gaussian kernel.
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)
plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
plt.axis([-4, 8, -0.2, 5])
plt.savefig('../scikit-learn-examples-figures/kernel-density-estimation-5--motivation-5--smoothed-out-blocks.svg')
plt.close()

# # # kernel density estimation in practice

# kernel: specifies the shape of the distribution placed at each point;
# kernel bandwidth: controls the size of the kernel at each point.
# The Scikit-Learn KDE implementation supports six kernels.
# Also in SciPy and statsmodels, but Scikit-Learn is preferred because of efficiency and flexibility.

# Instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])
# score_samples returns the log of the probability density
log_prob = kde.score_samples(x_d[:, None])
plt.fill_between(x_d, np.exp(log_prob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)
plt.savefig('../scikit-learn-examples-figures/kernel-density-estimation-6--kde.svg')
plt.close()

# # # selecting the bandwidth via cross-validation

# - too narrow a bandwidth -> a high-variance estimate (i.e., overfitting)
# -> the presence or absence of a single point makes a large difference;
#
# - too wide a bandwidth -> a high-bias estimate (i.e., underfitting)
# -> the structure in the data is washed out by the wide kernel.

# Use GridSearchCV to optimize the bandwidth
bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(
    KernelDensity(kernel='gaussian'),
    {'bandwidth': bandwidths},
    cv=LeaveOneOut()
)
grid.fit(x[:, None])

# Get a bandwidth that maximizes the score
# (which in this case defaults to the log-likelihood)
grid.best_params_
# {'bandwidth': 1.1233240329780276}

# Instantiate the KDE model with the optimized bandnwidth
kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
kde.fit(x[:, None])
# score_samples returns the log of the probability density
log_prob = kde.score_samples(x_d[:, None])
plt.fill_between(x_d, np.exp(log_prob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)
plt.savefig('../scikit-learn-examples-figures/kernel-density-estimation-7--kde-with-optimized-bandwidth.svg')
plt.close()
