# KNN Algorithm

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data  # Features
Y = iris.target # Labels


# Splitting the dataset
X_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Initialize the KNN classifies
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN model trained successfully")

# Make Predictions
y_pred = knn.predict(X_test)

# Evaluate the Model
# evaluate how well the model performed by comparing the predicted labels (y_pred) with the actual labels (y_test)
accuracy = accuracy_score(y_pred, Y_test)
print(f"Model accuracy : {accuracy:.4f}")

