import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

plt.style.use('seaborn-v0_8-whitegrid')

digits = load_digits()

# 8 x 8–pixel images -> the data is 64-dimensional.

# Apply PCA to the handwritten digits data
pca = PCA(2)  # Project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
# (1797, 64)
print(projected.shape)
# (1797, 2)

cmap = matplotlib.colormaps.get_cmap("rainbow")
colors = cmap(np.arange(cmap.N))
cmap = LinearSegmentedColormap.from_list("rainbow", colors, N=10)

plt.scatter(
    projected[:, 0],
    projected[:, 1],
    c=digits.target,
    edgecolor='none',
    alpha=0.5,
    cmap=cmap
)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.savefig('../scikit-learn-examples-figures/pca-for-classifying-handwritten-digits-1.svg')
plt.close()

# # # choosing the number of components

# cumulative explained variance ratio

pca = PCA().fit(digits.data)
# How much of the total 64-dimensional variance is contained within the first N components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('../scikit-learn-examples-figures/pca-for-classifying-handwritten-digits-2--cumsum--explained_variance_ratio_.svg')
plt.close()

# # # PCA as noise filtering


# Plot several of the input noise-free input samples
def plot_digits(data):
    fig, axes = plt.subplots(
        nrows=4,
        ncols=10,
        figsize=(10, 4),
        subplot_kw={
            'xticks': [],
            'yticks': []
        },
        gridspec_kw=dict(hspace=0.1, wspace=0.1)
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            data[i].reshape(8, 8),
            cmap='binary',
            interpolation='nearest',
            clim=(0, 16)
        )


plot_digits(digits.data)
plt.savefig('../scikit-learn-examples-figures/pca-for-classifying-handwritten-digits-3--noise-free-samples.svg')
plt.close()

# Add some random noise
rng = np.random.default_rng(42)
# some noise to 10
rng.normal(10, 1)
#  10.30471707975443
rng.normal(10, 2)
# 10.609434159508863
rng.normal(10, 3)
# 10.914151239263294
rng.normal(10, 4)
# 11.218868319017725

noisy = rng.normal(digits.data, 4)
plot_digits(noisy)
plt.savefig('../scikit-learn-examples-figures/pca-for-classifying-handwritten-digits-4--noisy-samples.svg')
plt.close()

# Train a PCA model on the noisy data, requesting
# that the projection preserve 50% of the variance
pca = PCA(0.50).fit(noisy)
pca.n_components_
# 12

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)
plt.savefig('../scikit-learn-examples-figures/pca-for-classifying-handwritten-digits-5--noise-filtered-samples.svg')
plt.close()
