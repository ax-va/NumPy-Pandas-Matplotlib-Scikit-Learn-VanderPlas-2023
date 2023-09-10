import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

digits = load_digits()
digits.data.shape
# (1797, 64)
# 1797 pictures of size of 8x8 of handwritten digits

kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape
# (10, 64)
# 10 clusters in 64 dimensions

# cluster centers learned by k-means
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.savefig('../scikit-learn-examples-figures/k-means-on-handwritten-digits-1--cluster-centers.svg')
plt.close()

clusters
# array([1, 2, 2, ..., 2, 3, 3], dtype=int32)
clusters.shape
# (1797,)

# Make the clusters to correspond to the targets (digits)
labels = np.zeros_like(clusters)
# array([0, 0, 0, ..., 0, 0, 0], dtype=int32)
labels.shape
# (1797,)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask], keepdims=True)[0]

# accuracy
accuracy_score(digits.target, labels)
# 0.7885364496382861

# confusion matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    cmap='Blues',
    xticklabels=digits.target_names,
    yticklabels=digits.target_names
)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('../scikit-learn-examples-figures/k-means-on-handwritten-digits-2--confusion-matrix.svg')
plt.close()

# t-distributed stochastic neighbor embedding algorithm (t-SNE)
# to preprocess the data before using k-means

# Project the data: this step will take several seconds
tsne = TSNE(n_components=2, init='random', learning_rate='auto',random_state=0)
digits_proj = tsne.fit_transform(digits.data)
# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
clusters = kmeans.fit_predict(digits_proj)
# Permute the labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask], keepdims=True)[0]
# Compute the accuracy
accuracy_score(digits.target, labels)
# 0.8597662771285476

# confusion matrix with preprocessing by t-SNE
mat = confusion_matrix(digits.target, labels)
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    cmap='Blues',
    xticklabels=digits.target_names,
    yticklabels=digits.target_names
)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('../scikit-learn-examples-figures/k-means-on-handwritten-digits-3--confusion-matrix-with-preprocessing-by-TSNE.svg')
plt.close()
