import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
# Load images of the digits 0 through 5 and visualize several of them
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])
plt.savefig('../matplotlib-examples-figures/handwritten-digits-1--samples.svg')
plt.close()

# dimensionality reduction from 64 (64 pixels) to 2

# Project the digits into 2 dimensions using Isomap
from sklearn.manifold import Isomap
iso = Isomap(n_components=2, n_neighbors=15)
projection = iso.fit_transform(digits.data)

cmap = matplotlib.colormaps.get_cmap("plasma")
colors = cmap(np.arange(cmap.N))
cmap = LinearSegmentedColormap.from_list("plasma", colors, N=6)

# Plot the results
plt.scatter(
    projection[:, 0], projection[:, 1],
    lw=0.1,
    c=digits.target,
    cmap=cmap
)
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)
plt.savefig('../matplotlib-examples-figures/handwritten-digits-2--dimensionality-reduction.svg')
plt.close()
# 2s and 3s are difficult to distinguish.
# 0s and 1s are better to distinguish.
