# Implementation of Decision Tree Classifier using scikit-learn

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Create a Dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(df)

#*-------------------------- DECISION TREE CLASSIFIER -----------------------------*
# Split the dataset into features and target variable
X = df[['sepal length (cm)', 'sepal width (cm)']]
Y = iris.target # model target should be the labels (iris.target) not petal features.

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5, random_state=42)

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the classifier
dt_classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy of Decision Tree Classifier: {accuracy * 100:.2f}%')

# Visualize the Decision Tree
plt.figure(figsize=(20,6))
tree.plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=iris.target_names)
plt.title('Decision Tree Visualization')
plt.show()
#*---------------------------------------------------------------------------------*

print("DECISION TREE REGRESSION")

#*-------------------------- DECISION TREE REGRESSION  -----------------------------*
# Initialize the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)

# Train the regressor
dt_regressor.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_reg = dt_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred_reg)
r2 = r2_score(Y_test, Y_pred_reg)
print(f'Mean Squared Error of Decision Tree Regressor: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')
#*---------------------------------------------------------------------------------*

# Visualize the Decision Tree Regressor
plt.figure(figsize=(20,6))
tree.plot_tree(dt_regressor, filled=True, feature_names=X.columns)
plt.title('Decision Tree Regressor Visualization')
plt.show()