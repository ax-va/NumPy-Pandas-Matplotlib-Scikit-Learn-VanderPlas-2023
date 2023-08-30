import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

digits = load_digits()
print(digits.keys())
# dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])

# Set up the figure
fig = plt.figure(figsize=(6, 6)) # figure size in inches
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1,
    hspace=0.05, wspace=0.05
)

# Plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap="binary", interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
plt.savefig('../scikit-learn-examples-figures/random-forest-for-classifying-handwritten-digits-1--data-with-labels.svg')
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    random_state=0
)

model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(metrics.classification_report(y_pred, y_test))
#               precision    recall  f1-score   support
#
#            0       1.00      0.97      0.99        38
#            1       1.00      0.96      0.98        45
#            2       0.95      1.00      0.98        42
#            3       0.98      0.98      0.98        45
#            4       0.97      1.00      0.99        37
#            5       0.98      0.98      0.98        48
#            6       1.00      1.00      1.00        52
#            7       1.00      0.96      0.98        50
#            8       0.94      0.98      0.96        46
#            9       0.98      0.98      0.98        47
#
#     accuracy                           0.98       450
#    macro avg       0.98      0.98      0.98       450
# weighted avg       0.98      0.98      0.98       450

# confusion matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('../scikit-learn-examples-figures/random-forest-for-classifying-handwritten-digits-2--confusion-matrix.svg')
plt.close()