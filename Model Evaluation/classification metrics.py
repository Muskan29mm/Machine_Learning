# Classification Metrics

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import weighted
from sklearn import datasets
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.20)

# Model train
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

print()

# Precision and Recall
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))

# F1 score
print("F1 score:", f1_score(y_test, y_pred, average="weighted"))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2])
cm_display.plot()
plt.show()

# AUC and ROC curve
y_true = [1, 0, 0, 1]
y_pred = [1, 0, 0.9, 0.2]
auc = np.round(roc_auc_score(y_true, y_pred), 3)
print("AUC:", auc)