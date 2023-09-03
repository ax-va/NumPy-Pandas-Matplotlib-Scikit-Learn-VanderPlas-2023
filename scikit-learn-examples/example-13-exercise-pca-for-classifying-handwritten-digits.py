import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

plt.style.use('seaborn-v0_8-whitegrid')

digits = load_digits()

# 8 x 8–pixel images -> the data is 64-dimensional.

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
plt.savefig('../scikit-learn-examples-figures/pca-for-classifying-handwritten-digits.svg')
plt.close()
