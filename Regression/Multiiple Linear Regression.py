# Multiple Linear Regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Loading datasets
california_housing = fetch_california_housing()
print(california_housing)

X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.Series(california_housing.target)

#  Selecting Features for Visualization
# Choose two features MedInc (median income) and AveRooms (average rooms) to simplify visualization in two dimensions
X = X[['MedInc', 'AveRooms']]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Initializing and training the model(Generating the model)
model = LinearRegression()
model.fit(X_train, Y_train)

print()

# Making Predictions
# Use the trained model to predict house prices on test data
y_pred = model.predict(X_test)
print(y_pred)

print()

# Evaluation
MSE = mean_squared_error(Y_test, y_pred)
print(MSE)

