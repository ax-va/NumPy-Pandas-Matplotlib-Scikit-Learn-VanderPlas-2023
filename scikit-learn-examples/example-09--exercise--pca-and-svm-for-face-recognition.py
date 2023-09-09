import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
# ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush'
#  'Gerhard Schroeder' 'Hugo Chavez' 'Junichiro Koizumi' 'Tony Blair']
print(faces.images.shape)
# (1348, 62, 47)


fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(8, 6))
ax
# array([[<Axes: >, <Axes: >, <Axes: >, <Axes: >, <Axes: >],
#        [<Axes: >, <Axes: >, <Axes: >, <Axes: >, <Axes: >],
#        [<Axes: >, <Axes: >, <Axes: >, <Axes: >, <Axes: >]], dtype=object)
ax.flat
#  <numpy.flatiter at 0x557b615f8700>
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(
        xticks=[],
        yticks=[],
        xlabel=faces.target_names[faces.target[i]]
    )
plt.savefig('../scikit-learn-examples-figures/pca-and-svm-for-face-recognition-1--faces.svg')
plt.close()

# Each image contains 62 × 47 pixels.

# Extract with PCA 150 fundamental components to feed into the support vector machine classifier
pca = PCA(n_components=150, whiten=True, svd_solver='randomized', random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
# pipeline
model = make_pipeline(pca, svc)

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(
    faces.data, faces.target, random_state=42
)

# Use grid search cross-validation to explore combinations of parameters.
# Adjust C (which controls the margin hardness) and
# gamma (which controls the size of the radial basis function kernel)
param_grid = {
    'svc__C': [1, 5, 10, 50],
    'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]
}
grid = GridSearchCV(model, param_grid)
grid.fit(X_train, y_train)
# %time grid.fit(X_train, y_train)
# CPU times: user 2min 12s, sys: 2min 48s, total: 5min 1s
# Wall time: 40.3 s
print(grid.best_params_)
# {'svc__C': 5, 'svc__gamma': 0.001}

# Choose the best model from the grid
model = grid.best_estimator_
y_fit = model.predict(X_test)

# some images and predicted labels
fig, ax = plt.subplots(nrows=4, ncols=6)
for i, axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(
        faces.target_names[y_fit[i]].split()[-1],
        color='black' if y_fit[i] == y_test[i] else 'red'
    )
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
plt.savefig('../scikit-learn-examples-figures/pca-and-svm-for-face-recognition-2--predicted-labels.svg')
plt.close()

# classification_report
print(classification_report(y_test, y_fit, target_names=faces.target_names))
#                    precision    recall  f1-score   support
#
#      Ariel Sharon       0.65      0.87      0.74        15
#      Colin Powell       0.83      0.88      0.86        68
#   Donald Rumsfeld       0.70      0.84      0.76        31
#     George W Bush       0.97      0.80      0.88       126
# Gerhard Schroeder       0.76      0.83      0.79        23
#       Hugo Chavez       0.93      0.70      0.80        20
# Junichiro Koizumi       0.86      1.00      0.92        12
#        Tony Blair       0.82      0.98      0.89        42
#
#          accuracy                           0.85       337
#         macro avg       0.82      0.86      0.83       337
#      weighted avg       0.86      0.85      0.85       337

# confusion_matrix
mat = confusion_matrix(y_test, y_fit)
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    cmap='Blues',
    xticklabels=faces.target_names,
    yticklabels=faces.target_names
)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.savefig('../scikit-learn-examples-figures/pca-and-svm-for-face-recognition-3--confusion-matrix.svg')
plt.close()
