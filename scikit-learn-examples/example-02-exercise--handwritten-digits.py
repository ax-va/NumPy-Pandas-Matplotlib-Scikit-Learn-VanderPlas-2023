import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

digits = load_digits()
digits.images.shape
# (1797, 8, 8)

# Visualize the first hundred of these
fig, axes = plt.subplots(
    nrows=10, ncols=10, figsize=(8, 8),
    subplot_kw={'xticks': [], 'yticks': []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1)
)
for i, ax in enumerate(axes.flat):
    ax.imshow(
        digits.images[i],
        cmap='binary',
        interpolation='nearest'
    )
    ax.text(
        0.05, 0.05,
        str(digits.target[i]),
        transform=ax.transAxes,
        color='green'
)
plt.savefig('../scikit-learn-examples-figures/handwritten-digits-1--data-with-labels.svg')
plt.close()

X = digits.data
X.shape
# (1797, 64)

y = digits.target
y.shape
# (1797,)

# # # unsupervised learning example: dimensionality reduction

# manifold learning algorithm called Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape
# (1797, 2)

cmap = matplotlib.colormaps.get_cmap("viridis")
colors = cmap(np.arange(cmap.N))
cmap = LinearSegmentedColormap.from_list(
    name="viridis",
    colors=colors,
    N=10  # 10 digits
)

# Isomap embedding of the digits data
plt.scatter(
    data_projected[:, 0], data_projected[:, 1],
    c=digits.target,
    edgecolor='none',
    alpha=0.5,
    cmap=cmap
)
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig('../scikit-learn-examples-figures/handwritten-digits-2--Isomap.svg')
plt.close()

# # # classification on digits

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = GaussianNB()
model.fit(X_train, y_train)
y_model = model.predict(X_test)
accuracy_score(y_test, y_model)
# 0.8333333333333334

# confusion matrix
mat = confusion_matrix(y_test, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False, cmap='Blues')
plt.savefig('../scikit-learn-examples-figures/handwritten-digits-3--confusion-matrix.svg')
plt.close()

# data with green for correct labels and red for incorrect labels
fig, axes = plt.subplots(
    nrows=10,
    ncols=10,
    figsize=(8, 8),
    subplot_kw={'xticks': [], 'yticks': []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1)
)

test_images = X_test.reshape(-1, 8, 8)
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
    ax.text(
        0.05, 0.05,
        str(y_model[i]),
        transform=ax.transAxes,
        color='green' if (y_test[i] == y_model[i]) else 'red'
    )
plt.savefig('../scikit-learn-examples-figures/handwritten-digits-4--testing-data-with-predicted-labels.svg')
plt.close()
