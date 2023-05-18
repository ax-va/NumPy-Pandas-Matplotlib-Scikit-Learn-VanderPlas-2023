import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)

mean = [0, 0]
cov = [[1, 2],
       [2, 5]]

X = rng.multivariate_normal(mean, cov, 100)
print(X.shape)  # (100, 2)

plt.scatter(X[:, 0], X[:, 1])
plt.savefig('../figures/selecting-subset-of-matrix-rows-by-fancy-indexing-1.svg')
plt.close()

# Select 20 random indices with no repeat
indices = np.random.choice(X.shape[0], 20, replace=False)
selected = X[indices]  # fancy indexing
print(selected.shape)  # (20, 2)

plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selected[:, 0], selected[:, 1], facecolor='none', edgecolor='black', s=200)
plt.savefig('../figures/selecting-subset-of-matrix-rows-by-fancy-indexing-2.svg')
plt.close()
