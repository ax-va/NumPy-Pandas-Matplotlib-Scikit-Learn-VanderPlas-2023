import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.cluster import MiniBatchKMeans

china = load_sample_image("china.jpg")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)
plt.savefig('../scikit-learn-examples-figures/k-means-for-color-compression-1--china-image.jpg')
plt.close()

china.shape
# (427, 640, 3)
# That is (height, width, RGB)

china
# array([[[174, 201, 231],
#         [174, 201, 231],
#         [174, 201, 231],
#         ...,
#         [250, 251, 255],
#         [250, 251, 255],
#         [250, 251, 255]],
#
#        [[172, 199, 229],
#         [173, 200, 230],
#         [173, 200, 230],
#         ...,
#         [251, 252, 255],
#         [251, 252, 255],
#         [251, 252, 255]],
#
#        [[174, 201, 231],
#         [174, 201, 231],
#         [174, 201, 231],
#         ...,
#         [252, 253, 255],
#         [252, 253, 255],
#         [252, 253, 255]],
#
#        ...,
#
#        [[ 88,  80,   7],
#         [147, 138,  69],
#         [122, 116,  38],
#         ...,
#         [ 39,  42,  33],
#         [  8,  14,   2],
#         [  6,  12,   0]],
#
#        [[122, 112,  41],
#         [129, 120,  53],
#         [118, 112,  36],
#         ...,
#         [  9,  12,   3],
#         [  9,  15,   3],
#         [ 16,  24,   9]],
#
#        [[116, 103,  35],
#         [104,  93,  31],
#         [108, 102,  28],
#         ...,
#         [ 43,  49,  39],
#         [ 13,  21,   6],
#         [ 15,  24,   7]]], dtype=uint8)

data = china / 255.0  # to 0 to 1 scale
data = data.reshape(-1, 3)
data.shape
# (273280, 3)


def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    # choose a random subset
    rng = np.random.default_rng(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(title, size=20)


plot_pixels(data, title='Input color space: 16 million possible colors')
plt.savefig('../scikit-learn-examples-figures/k-means-for-color-compression-2--color-space.png')
plt.close()

# Reduce these 16 million colors to just 16 colors,
# using a k-means clustering across the pixel space

# mini-batch k-means
kmeans = MiniBatchKMeans(n_clusters=16, n_init="auto")
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
plot_pixels(data, colors=new_colors, title="Reduced color space: 16 colors")
plt.savefig('../scikit-learn-examples-figures/k-means-for-color-compression-3--reduced-color-space.png')
plt.close()

# comparison of the full-color image (left) and the 16-color image (right)
china_recolored = new_colors.reshape(china.shape)
fig, ax = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(16, 6),
    subplot_kw=dict(xticks=[], yticks=[])
)
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16)
plt.savefig('../scikit-learn-examples-figures/k-means-for-color-compression-4--original-and-recolored-images.png')
plt.close()
# The fidelity does not correspond to JPEG.
