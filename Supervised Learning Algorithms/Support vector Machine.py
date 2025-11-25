# Implementation of Support Vector Machine (SVM) Classifier using scikit-learn

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the Digits dataset
digits = load_digits()

df = pd.DataFrame(data=digits.data)
print(df)

# Split the dataset into features and target variable
X = df
Y = digits.target

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Initialize the SVM Classifier
svm_classifier = SVC(kernel='linear', random_state=42)
# Train the classifier
svm_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy of SVM Classifier: {accuracy * 100:.2f}%')

# Visualize the results

# scale -> PCA 2D
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=2).fit(X_train_s)
X_train_pca = pca.transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

# retrain an SVM on the 2D projection with the same important params as original
orig_params = svm_classifier.get_params()
svm_pca = SVC(**{k: orig_params[k] for k in ['kernel', 'degree', 'gamma', 'C', 'class_weight', 'probability', 'random_state'] if k in orig_params})
svm_pca.fit(X_train_pca, Y_train)
Y_pred_pca = svm_pca.predict(X_test_pca)

# decision surface grid
x_min, x_max = X_train_pca[:,0].min() - 1, X_train_pca[:,0].max() + 1
y_min, y_max = X_train_pca[:,1].min() - 1, X_train_pca[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# plot decision regions + test points (true labels) + misclassified highlights
plt.figure(figsize=(14,6))

ax = plt.subplot(1,2,1)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
scatter = ax.scatter(X_test_pca[:,0], X_test_pca[:,1], c=Y_test, cmap='tab10', edgecolor='k', s=40)
miss = Y_test != Y_pred_pca
if miss.any():
    ax.scatter(X_test_pca[miss,0], X_test_pca[miss,1], facecolors='none', edgecolors='r', s=120, linewidths=1.5, label='misclassified')
ax.set_title('SVM decision regions (PCA 2D)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(loc='upper right', markerscale=0.6)

# confusion matrix of the PCA-trained SVM on test set
ax2 = plt.subplot(1,2,2)
cm = confusion_matrix(Y_test, Y_pred_pca)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
ax2.set_title('Confusion matrix (PCA SVM)')

plt.tight_layout()
plt.show()


# Changing kernel types and evaluating
# kernel = 'rbf'  # change to 'linear', 'poly', 'sigmoid', etc.
# svc = SVC(kernel=kernel, gamma='scale', random_state=42, degree=3)  # degree used for 'poly'
# svc.fit(X_train, Y_train)
# y_pred = svc.predict(X_test)
# print(f"Accuracy ({kernel}): {accuracy_score(Y_test, y_pred):.4f}")


