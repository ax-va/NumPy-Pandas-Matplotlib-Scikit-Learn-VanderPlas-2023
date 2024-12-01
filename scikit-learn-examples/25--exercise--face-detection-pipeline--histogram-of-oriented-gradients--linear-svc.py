from itertools import chain
from typing import Generator
import matplotlib.pyplot as plt
import numpy as np
# the installed scikit-image package
import skimage.data
from skimage import data, color, feature, transform
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


plt.style.use('seaborn-v0_8-whitegrid')

# linear support vector classification (Linear SVC)

# # # HOG features

# Histogram of oriented gradients (HOG):
# - capturing edge, contour, and texture information;
# - suppression of the effect of illumination across the image.

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualize=True)
fig, ax = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6),
    subplot_kw=dict(xticks=[], yticks=[])
)
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')
ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features')
plt.savefig('../scikit-learn-examples-figures/face-detection-pipeline-1--histogram-of-oriented-gradients.png')
plt.close()

# # # HOG in action: a simple face detector

# 1. Obtain a set of image thumbnails of faces to constitute "positive" training samples;
# 2. Obtain a set of image thumbnails of non-faces to constitute "negative" training samples;
# 3. Extract HOG features from these training samples;
# 4. Train a linear SVM classifier on these samples;
# 5. For an "unknown" image, pass a sliding window across the image,
# using the model to evaluate whether that window contains a face or not;
# 6. If detections overlap, combine them into a single window.

# # # 1. Obtain a set of image thumbnails of faces to constitute "positive" training samples
faces = fetch_lfw_people()
positive_patches = faces.images
positive_patches.shape
# (13233, 62, 47)
# 13233 images that contains faces

# # # 2. Obtain a set of image thumbnails of non-faces to constitute "negative" training samples
data.camera().shape
# (512, 512)

imgs_to_use = [
    'camera',
    'text',
    'coins',
    'moon',
    'page',
    'clock',
    'immunohistochemistry',
    'chelsea',
    'coffee',
    'hubble_deep_field'
]

raw_images = (getattr(data, name)() for name in imgs_to_use)
images = [color.rgb2gray(image) if image.ndim == 3 else image for image in raw_images]


def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(
        patch_size=extracted_patch_size,
        max_patches=N,
        random_state=0
    )
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])
    return patches


negative_patches = np.vstack([extract_patches(im, 1000, scale) for im in images for scale in [0.5, 1.0, 2.0]])
negative_patches.shape
# (30000, 62, 47)
# 30_000 images that contain no faces

fig, ax = plt.subplots(nrows=6, ncols=10)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500 * i], cmap='gray')
    axi.axis('off')
plt.savefig('../scikit-learn-examples-figures/face-detection-pipeline-2--negative-patches-for-hog.png')
plt.close()

# # # 3. Extract HOG features from these training samples

"""
p1 ="a"
p2 = "bcd"
chain(p1, p2)
# <itertools.chain at 0x7fb2875664a0>
list(chain(p1, p2))
# ['a', 'b', 'c', 'd']
p3 = ["aa", "bb"]
p4 = "cc"
list(chain(p3, p4))
# ['aa', 'bb', 'c', 'c']
"""

X_train = np.array(
    [
        feature.hog(img) for img in chain(positive_patches, negative_patches)
    ]
)
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1
X_train.shape
# (43233, 1215)

# # # 4. Train a linear SVM classifier on these samples;

# high-dimensional binary classification task -> linear support vector machine

# Use a simple Gaussian naive Bayes estimator to get a quick baseline
cross_val_score(GaussianNB(), X_train, y_train)
# array([0.95663236, 0.972476  , 0.97363247, 0.97640527, 0.97536433])
# accuracy larger than 95% for training data

# Use linear support vector classification
grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0], "dual": ["auto"]})
grid.fit(X_train, y_train)
grid.best_score_
# 0.9887816989072598
# accuracy up to nearly 99% for training data
grid.best_params_
# {'C': 2.0, 'dual': 'auto'}

# Take the model with the best estimator
model = grid.best_estimator_
model.fit(X_train, y_train)
# LinearSVC(C=2.0, dual='auto')

# # # 5. For an "unknown" image, pass a sliding window across the image,
# # # using the model to evaluate whether that window contains a face or not

# Take the astronaut image to test
test_image = skimage.data.astronaut()
test_image = skimage.color.rgb2gray(test_image)
test_image = skimage.transform.rescale(test_image, scale=0.5)
test_image = test_image[:160, 40:180]
test_image.shape
# (160, 140)

px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
fig = plt.figure(
    frameon=False,
    figsize=(test_image.shape[0] * px, test_image.shape[1] * px)
)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.imshow(test_image, cmap='gray')
fig.add_axes(ax)
ax.axis('off')
plt.savefig('../scikit-learn-examples-figures/face-detection-pipeline-3--test-image-to-locate-face.png')
plt.close()

# Create a window that iterates over patches of this image,
# and compute HOG features for each patch


def sliding_window(img, patch_size=positive_patches[0].shape, i_step=2, j_step=2, scale=1.0):
    """
    Args:
        img: image
        patch_size: height and width of the image
        i_step: pixels to change the sliding window in height
        j_step: pixels to change the sliding window in width
        scale: value to scale the image
    Yields: patch indices, patch
    """
    N_i, N_j = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - N_i, i_step):
        for j in range(0, img.shape[1] - N_i, j_step):
            patch = img[i:i + N_i, j:j + N_j]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


positive_patches[0].shape
# (62, 47)

indices, patches = zip(*sliding_window(test_image))
type(patches[0])
# numpy.ndarray
patches_hog = np.array([feature.hog(patch) for patch in patches])
patches_hog.shape
# (1911, 1215)

labels = model.predict(patches_hog)
labels.sum()
# 46.0
# 46 detections found in nearly 2000 patches

# Show the detections
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')
N_i, N_j = positive_patches[0].shape
# (62, 47)
indices = np.array(indices)
for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle(xy=(j, i), width=N_j, height=N_i, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))
plt.savefig('../scikit-learn-examples-figures/face-detection-pipeline-4--face-detections.png')
plt.close()

# # # caveats and improvements

# - training set, especially for negative features, is not very complete
# -> the current model leads to many false detections in other regions of the *full* astronaut image;
# - current pipeline searches only at one scale
# -> resize each patch using skimage.transform.resize;
# - combine overlapped detection patches
# -> via an unsupervised clustering approach or
# via a procedural approach such as non-maximum suppression;
# - pipeline should be streamlined
# - deep learning
