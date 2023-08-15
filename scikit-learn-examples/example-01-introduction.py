import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# # # data representation

iris = sns.load_dataset('iris')
iris.head()
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa

# features matrix = X
# the shape of the features matrix = [n_samples, n_features]
# label or target array = y

# the rows of the table = samples
# the number of rows = n_samples
# the columns of the table = features
# the number of columns = n_features

sns.pairplot(iris, hue='species', height=1.5)
plt.savefig('../scikit-learn-examples-figures/introduction-1--iris-dataset.svg')
plt.close()

X_iris = iris.drop('species', axis=1)
X_iris.shape
# (150, 4)

y_iris = iris['species']
y_iris.shape
# (150,)

# # # the estimator API

# # # supervised learning example: simple linear regression

# training points
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)

model = LinearRegression(fit_intercept=True)
# LinearRegression()

# Make the features matrix of size [n_samples, n_features]
X = x[:, np.newaxis]
X.shape
# (50, 1)

model.fit(X, y)
# LinearRegression()

# model parameters: slope and intercept close to the ones of 2 and -1
model.coef_
# array([1.9776566])
model.intercept_
# -0.9033107255311146

x_fit = np.linspace(-1, 11)
x_fit.shape
# (50,)
# Make the features matrix of size [n_samples, n_features]
X_fit = x_fit[:, np.newaxis]
X_fit.shape
# (50, 1)
# Fit the model
y_fit = model.predict(X_fit)

# Plot the training points
plt.scatter(x, y)
# Plot the fitting line
plt.plot([x_fit[0], x_fit[-1]], [y_fit[0], y_fit[-1]])
plt.savefig('../scikit-learn-examples-figures/introduction-2--simple-linear-regression.svg')
plt.close()

# # # supervised learning example: Iris classification

# # # Gaussian naive Bayes

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)

model = GaussianNB()
model.fit(X_train, y_train)
y_model = model.predict(X_test)
accuracy_score(y_test, y_model)
# 0.9736842105263158

# # # unsupervised learning example: Iris dimensionality

# # # principal component analysis (PCA)

model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)

iris['PCA axis 0'] = X_2D[:, 0]
iris['PCA axis 1'] = X_2D[:, 1]
sns.lmplot(
    x="PCA axis 0",
    y="PCA axis 1",
    hue='species',
    data=iris,
    fit_reg=False
)
plt.savefig('../scikit-learn-examples-figures/introduction-3--PCA.svg')
plt.close()

# # # unsupervised learning example: Iris clustering

# # # Gaussian mixture model (GMM)

model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)
iris['cluster'] = y_gmm
sns.lmplot(
    x="PCA axis 0",
    y="PCA axis 1",
    data=iris,
    hue='species',
    col='cluster',
    fit_reg=False
)
# k-means clusters
plt.savefig('../scikit-learn-examples-figures/introduction-4--GMM-k-means-clusters.svg')
plt.close()
