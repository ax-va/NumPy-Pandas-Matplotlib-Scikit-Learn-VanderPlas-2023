import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

plt.style.use('seaborn-v0_8-whitegrid')

# # # Gaussian naive Bayes

# Data from each label is drawn
# from a simple Gaussian distribution
# with no covariance between dimensions.

# Generate training data in the form of isotropic Gaussian blobs for clustering
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.savefig('../scikit-learn-examples-figures/naive-bayes-1--gaussian-naive-bayes-1--training-data.svg')
plt.close()

model = GaussianNB()
model.fit(X, y)

# new data to predict the labels
rng = np.random.RandomState(0)
X_new = [-6, -14] + [14, 18] * rng.rand(2000, 2)
y_new = model.predict(X_new)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(X_new[:, 0], X_new[:, 1], c=y_new, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.savefig('../scikit-learn-examples-figures/naive-bayes-2--gaussian-naive-bayes-2--training-data-and-new-data.svg')
plt.close()

y_prob = model.predict_proba(X_new)
y_prob[-8:].round(2)
# array([[0.89, 0.11],
#        [1.  , 0.  ],
#        [1.  , 0.  ],
#        [1.  , 0.  ],
#        [1.  , 0.  ],
#        [1.  , 0.  ],
#        [0.  , 1.  ],
#        [0.15, 0.85]])

# The columns give the posterior probabilities
# of the first and second labels of the last eight points, respectively.

# # # multinomial naive Bayes on an example of text classification

data = fetch_20newsgroups()
data.target_names
# ['alt.atheism',
#  'comp.graphics',
#  'comp.os.ms-windows.misc',
#  'comp.sys.ibm.pc.hardware',
#  'comp.sys.mac.hardware',
#  'comp.windows.x',
#  'misc.forsale',
#  'rec.autos',
#  'rec.motorcycles',
#  'rec.sport.baseball',
#  'rec.sport.hockey',
#  'sci.crypt',
#  'sci.electronics',
#  'sci.med',
#  'sci.space',
#  'soc.religion.christian',
#  'talk.politics.guns',
#  'talk.politics.mideast',
#  'talk.politics.misc',
#  'talk.religion.misc']

# some categories
categories = [
    'talk.religion.misc',
    'soc.religion.christian',
    'sci.space',
    'comp.graphics'
]
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

print(train.data[5][48:])
# Subject: Federal Hearing
# Originator: dmcgee@uluhe
# Organization: School of Ocean and Earth Science and Technology
# Distribution: usa
# Lines: 10
#
#
# Fact or rumor....?  Madalyn Murray O'Hare an atheist who eliminated the
# use of the bible reading and prayer in public schools 15 years ago is now
# going to appear before the FCC with a petition to stop the reading of the
# Gospel on the airways of America.  And she is also campaigning to remove
# Christmas programs, songs, etc from the public schools.  If it is true
# then mail to Federal Communications Commission 1919 H Street Washington DC
# 20054 expressing your opposition to her request.  Reference Petition number
#
# 2493.

# TfidfVectorizer is used to get the term frequency–inverse document frequency (TF–IDF)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

mat = confusion_matrix(test.target, labels)
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    xticklabels=train.target_names,
    yticklabels=train.target_names,
    cmap='Blues'
)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.savefig('../scikit-learn-examples-figures/naive-bayes-3--multinomial-naive-bayes--confusion-matrix-for-text-classification.svg')
plt.close()


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


predict_category('sending a payload to the ISS')
# 'sci.space'

predict_category('discussing the existence of God')
# 'soc.religion.christian'

predict_category('determining the screen resolution')
# 'comp.graphics'

# The NB classification usually works as well as or better than more complicated classifiers:
# - if the naive assumptions actually match the data (very rare in practice)
# - for very well-separated categories, when model complexity is less important
# - for very high-dimensional data, when model complexity is less important
