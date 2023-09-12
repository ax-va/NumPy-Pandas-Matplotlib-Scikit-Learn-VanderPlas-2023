import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-whitegrid')

# Goal: generate new handwritten digits

digits = load_digits()
digits.data.shape
# (1797, 64)


def plot_digits(data):
    fig, ax = plt.subplots(
        nrows=5,
        ncols=10,
        figsize=(8, 4),
        subplot_kw=dict(xticks=[], yticks=[])
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


plot_digits(digits.data)
plt.savefig('../scikit-learn-examples-figures/gmms-for-generating-new-handwritten-digits-1--data.svg')
plt.close()

# Use PCA and preserve 99% of the variance in the projected data
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
data.shape
# (1797, 41)

# Use AIC for choosing the appropriate number of GMM components
n_components = np.arange(50, 210, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0) for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics)
plt.savefig('../scikit-learn-examples-figures/gmms-for-generating-new-handwritten-digits-2--aic.svg')
plt.close()
# Choose 140

gmm = GaussianMixture(n_components=140, covariance_type='full', random_state=0)
gmm.fit(data)
gmm.converged_
# True

# Draw samples of 100 new points
# within this 41-dimensional projected space,
# using the GMM as a generative model
data_new, label_new = gmm.sample(100)
data_new.shape
# (100, 41)

# Use the inverse transform of the PCA object to construct the new digits
digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)
plt.savefig('../scikit-learn-examples-figures/gmms-for-generating-new-handwritten-digits-3--new-generated-data.svg')
plt.close()