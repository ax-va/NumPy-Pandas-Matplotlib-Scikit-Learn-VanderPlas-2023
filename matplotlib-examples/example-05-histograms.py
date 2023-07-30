from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

rng = np.random.default_rng(1701)
data = rng.normal(size=1000)
plt.hist(data)
plt.savefig('../matplotlib-examples-figures/histograms-1--histogram-1.svg')
plt.close()

plt.hist(
    data,
    bins=30,
    density=True,
    alpha=0.5,
    histtype='stepfilled',
    color='steelblue',
    edgecolor='none'
)
plt.savefig('../matplotlib-examples-figures/histograms-2--histogram-2.svg')
plt.close()

# histograms of several distributions
x1 = rng.normal(0, 0.8, 1000)
x2 = rng.normal(-2, 1, 1000)
x3 = rng.normal(3, 2, 1000)
kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)
plt.savefig('../matplotlib-examples-figures/histograms-3--several-distributions.svg')
plt.close()

# only computing histograms by NumPy
counts, bin_edges = np.histogram(data, bins=5)
# the number of points in a given bin
counts
# array([23, 241, 491, 224, 21])
bin_edges
# array([-3.26668015, -1.94720849, -0.62773683,  0.69173484,  2.0112065 ,
#         3.33067816])

# two-dimensional histograms and binnings

# multivariate Gaussian distribution
mean = [0, 0]
cov = [[1, 1], [1, 2]]
mvn = rng.multivariate_normal(mean, cov, 10000)
# array([[-1.73545263, -2.66151433],
#        [ 1.08103906,  2.23475518],
#        [ 0.13872987, -0.7636978 ],
#        ...,
#        [ 0.98475794,  0.49041169],
#        [-0.30050561, -0.28890484],
#        [-0.50254472,  0.36890102]])
x, y = mvn.T

# two-dimensional histogram

plt.hist2d(x, y, bins=30)
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.savefig('../matplotlib-examples-figures/histograms-4--two-dimensional-histogram.svg')
plt.close()

# only computing histograms by NumPy
counts, xedges, yedges = np.histogram2d(x, y, bins=30)
print(counts.shape)
# (30, 30)

# For more than two dimensions, see the np.histogramdd function.

# hexagonal binnings

plt.hexbin(x, y, gridsize=30)
cb = plt.colorbar(label='count in bin')
plt.savefig('../matplotlib-examples-figures/histograms-5--hexagonal-binning.svg')
plt.close()

# kernel density estimation (KDE)

# Fit an array of size [Ndim, Nsamples]
data = np.vstack([x, y])
# array([[-1.73545263,  1.08103906,  0.13872987, ...,  0.98475794,
#         -0.30050561, -0.50254472],
#        [-2.66151433,  2.23475518, -0.7636978 , ...,  0.49041169,
#         -0.28890484,  0.36890102]])
kde = gaussian_kde(data)

# Evaluate on a regular grid
x_grid = np.linspace(-3.5, 3.5, 40)
y_grid = np.linspace(-6, 6, 40)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z = kde.evaluate(np.vstack([X_grid.ravel(), Y_grid.ravel()]))

# Plot the result as an image
plt.imshow(
    Z.reshape(X_grid.shape),
    origin='lower',
    aspect='auto',
    extent=[-3.5, 3.5, -6, 6]
)
plt.colorbar().set_label("density")
plt.savefig('../matplotlib-examples-figures/histograms-6--kernel-density-estimation.svg')
plt.close()

# Other KDE implementations:
# - sklearn.neighbors.KernelDensity,
# - statsmodels.nonparametric.KDEMultivariate
