import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# # # categorical features

data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

# one-hot encoding

vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)
# array([[     0,      1,      0, 850000,      4],
#        [     1,      0,      0, 700000,      3],
#        [     0,      0,      1, 650000,      3],
#        [     1,      0,      0, 600000,      2]])

vec.get_feature_names_out()
# array(['neighborhood=Fremont', 'neighborhood=Queen Anne',
#        'neighborhood=Wallingford', 'price', 'rooms'], dtype=object)

# sparse input
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)
# <4x5 sparse matrix of type '<class 'numpy.int64'>'
#         with 12 stored elements in Compressed Sparse Row format>

# See also
# sklearn.preprocessing.OneHotEncoder,
# sklearn.feature_extraction.FeatureHasher

# # # text features

sample = [
    'problem of evil',
    'evil queen',
    'horizon problem'
]

# word counts

vec = CountVectorizer()
X = vec.fit_transform(sample)
# <3x5 sparse matrix of type '<class 'numpy.int64'>'
#         with 7 stored elements in Compressed Sparse Row format>

# Inspect with Pandas
pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
#    evil  horizon  of  problem  queen
# 0     1        0   1        1      0
# 1     1        0   0        0      1
# 2     0        1   0        1      0

# term frequency–inverse document frequency (TF–IDF)

vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
#        evil   horizon        of   problem     queen
# 0  0.517856  0.000000  0.680919  0.517856  0.000000
# 1  0.605349  0.000000  0.000000  0.000000  0.795961
# 2  0.000000  0.795961  0.000000  0.605349  0.000000

# # # image features

# https://scikit-image.org/

# # # derived features

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)
plt.savefig('../scikit-learn-examples-figures/feature-engineering-1--derived-features-1--data.svg')
plt.close()

# linear regression

X = x[:, np.newaxis]
# array([[1],
#        [2],
#        [3],
#        [4],
#        [5]])
model = LinearRegression().fit(X, y)
y_fit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, y_fit)
plt.savefig('../scikit-learn-examples-figures/feature-engineering-2--derived-features-2--linear-regression.svg')
plt.close()

# Add polynomial features to the data

poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
# array([[  1.,   1.,   1.],
#        [  2.,   4.,   8.],
#        [  3.,   9.,  27.],
#        [  4.,  16.,  64.],
#        [  5.,  25., 125.]])

model = LinearRegression().fit(X2, y)
y2_fit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, y2_fit)
plt.savefig('../scikit-learn-examples-figures/feature-engineering-3--derived-features-3--polynomial-features.svg')
plt.close()

# also later: basis function regression

# # # imputation of missing data

X = np.array([[np.nan, 0,      3],
              [3,      7,      9],
              [3,      5,      2],
              [4,      np.nan, 6],
              [8,      8,      1]])
y = np.array([14, 16, -1, 8, -5])

# imputation of missing values = replace the missing values with some appropriate fill value

# SimpleImputer is a baseline imputation approach using
# the mean, median, or most frequent value

imp = SimpleImputer(strategy='mean')  # mean of the remaining values in the column
X2 = imp.fit_transform(X)
# array([[4.5, 0. , 3. ],
#        [3. , 7. , 9. ],
#        [3. , 5. , 2. ],
#        [4. , 5. , 6. ],
#        [8. , 8. , 1. ]])

model = LinearRegression().fit(X2, y)
model.predict(X2)
# array([13.14869292, 14.3784627 , -1.15539732, 10.96606197, -5.33782027])

# # # feature pipelines

model = make_pipeline(
    SimpleImputer(strategy='mean'),
    PolynomialFeatures(degree=2),
    LinearRegression()
)
model.fit(X, y)  # X with missing values, from above
y
# array([14, 16, -1,  8, -5])
model.predict(X)  # training data = testing data
# array([14., 16., -1.,  8., -5.])