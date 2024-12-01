import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessRegressor

plt.style.use('seaborn-v0_8-whitegrid')

rng = np.random.default_rng(0)

x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black')
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-01--points.svg')
plt.close()


for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(
        rng.random(2), rng.random(2), marker,
        color='black',
        label=f"marker='{marker}'"
    )
plt.legend(numpoints=3, fontsize=13)
plt.xlim(0, 1.8)
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-02--markers.svg')
plt.close()

plt.plot(x, y, '-ok')
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-03--line-and-points.svg')
plt.close()

plt.plot(
    x, y, '-p',
    color='gray',
    markersize=15,
    linewidth=4,
    markerfacecolor='white',
    markeredgecolor='gray',
    markeredgewidth=2
)
plt.ylim(-1.2, 1.2)
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-04--line-and-marker-properties.svg')
plt.close()

# scatter plots with plt.scatter

plt.scatter(x, y, marker='o')
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-05--scatter.svg')
plt.close()

# plt.scatter can control individual points

x = rng.normal(size=100)
y = rng.normal(size=100)
colors = rng.random(100)  # 100 random floats in the half-open interval [0.0, 1.0)
sizes = 1000 * rng.random(100)  # in pixels
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3)
plt.colorbar()  # show color scale
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-06--scatter-colors-and-sizes.svg')
plt.close()

iris = load_iris()
features = iris.data.T
plt.scatter(
    features[0], features[1],
    alpha=0.4,
    s=100*features[3],
    c=iris.target,
    cmap='viridis'
)
# iris.feature_names[2]
# 'petal length (cm)'
# iris.feature_names[3]
# 'petal width (cm)'
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-07--scatter-for-iris-dataset.svg')
plt.close()

# plt.plot can be noticeably more efficient than plt.scatter for large datasets

# visualizing uncertainties

# basic errorbars

x = np.linspace(0, 10, 50)
dy = 0.5
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k')
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-08--errorbar.svg')
plt.close()

plt.errorbar(
    x, y,
    yerr=dy,
    fmt='o',
    color='black',
    ecolor='lightgray',
    elinewidth=3,
    capsize=0
)
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-09--errorbar-fine-tuning.svg')
plt.close()

# continuous errors

# simple Gaussian process regression

# Create a model and data
# define the model and draw some data
model = lambda t: t * np.sin(t)
xdata = np.array([1, 3, 5, 6, 8])
# array([1, 3, 5, 6, 8])
ydata = model(xdata)
# array([ 0.84147098,  0.42336002, -4.79462137, -1.67649299,  7.91486597])

# Compute the Gaussian process fit
gp = GaussianProcessRegressor()
# GaussianProcessRegressor()
xdata[:, np.newaxis]
# array([[1],
#        [3],
#        [5],
#        [6],
#        [8]])
gp.fit(xdata[:, np.newaxis], ydata)
# GaussianProcessRegressor()

xfit = np.linspace(0, 10, 1000)
yfit, dyfit = gp.predict(xfit[:, np.newaxis], return_std=True)

# Visualize the result
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')
plt.fill_between(
    xfit, yfit - dyfit, yfit + dyfit,
    color='gray', alpha=0.2
)
plt.xlim(0, 10)
plt.savefig('../matplotlib-examples-figures/simple-scatter-plots-10--fill_between.svg')
plt.close()
