# Linear Regression
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

disease = datasets.load_diabetes()
# print(disease)

print()

# Getting Labels
print(disease.keys())

print()

# Slice the data
disease_x = disease.data[:, np.newaxis,2]
print(disease_x)


print()

# Splitting the data
disease_x_train = disease_x[:-30]
disease_x_test = disease_x[-20:]

disease_y_train = disease.target[:-30]
disease_y_test = disease.target[-20:]

print()

# Generating the model
reg = linear_model.LinearRegression()
reg.fit(disease_x_train, disease_y_train)

y_predict = reg.predict(disease_x_test)
print(y_predict)

print()

# Evaluation
MSE = mean_squared_error(disease_y_test, y_predict)
print(MSE)
