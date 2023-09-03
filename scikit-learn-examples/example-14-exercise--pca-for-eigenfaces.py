import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people  # Labeled Faces in the Wild (LFW) dataset

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
# ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush'
#  'Gerhard Schroeder' 'Hugo Chavez' 'Junichiro Koizumi' 'Tony Blair']
print(faces.images.shape)
# (1348, 62, 47)

# Use the "random" eigensolver in the PCA estimator: faster, but less accuracy
pca = PCA(n_components=150, svd_solver='randomized', random_state=42)
pca.fit(faces.data)

fig, axes = plt.subplots(
    nrows=3,
    ncols=8,
    figsize=(9, 4),
    subplot_kw={
        'xticks': [],
        'yticks': []
    },
    gridspec_kw=dict(hspace=0.1, wspace=0.1)
)
# eigenfaces = eigenvectors
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
plt.savefig('../scikit-learn-examples-figures/pca-for-eigenfaces-1--eigenfaces.svg')
plt.close()

# cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('../scikit-learn-examples-figures/pca-for-eigenfaces-2--cumulative-explained-variance.svg')
plt.close()

# Compute the components and projected faces
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

# Plot the results
fig, ax = plt.subplots(
    nrows=2, ncols=10, figsize=(10, 2.5),
    subplot_kw={'xticks': [], 'yticks': []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1)
)
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction')
plt.savefig('../scikit-learn-examples-figures/pca-for-eigenfaces-3--full-dim-and-150-dim.svg')
plt.close()
