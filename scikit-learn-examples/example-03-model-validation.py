from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# # # model validation

# # # wrong way

iris = load_iris()
X = iris.data
y = iris.target

# k-nearest neighbors classifier:
# the label of an unknown point is the same
# as the label of its closest training point

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)
y_model = model.predict(X)
accuracy_score(y, y_model)
# 1.0

# It trains and evaluates the model on the same data.
# This model is an instance-based estimator that simply stores the training data.

# # # right way: holdout sets (testing data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=0,
    train_size=0.5
)

model.fit(X_train, y_train)
y_model = model.predict(X_test)
accuracy_score(y_test, y_model)
# 0.9066666666666666

# # # cross-validation

# # # two-fold cross-validation

X1 = X_train
X2 = X_test
y1 = y_train
y2 = y_test

y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)
# (0.96, 0.9066666666666666)

# # # five-fold cross-validation

cross_val_score(model, X, y, cv=5)
# array([0.96666667, 0.96666667, 0.93333333, 0.93333333, 1.        ])

# # # leave-one-out cross validation

# We train on all points but one in each trial
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
# array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

# 150 points -> 150 scores

# Estimate the error rate
scores.mean()
# 0.96


