# Cross Validation
# There are two methods in cross validation i.e 1. Hold-out method      2. K-fold cross validation

# Hold-out method
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y , test_size = 0.20, random_state=42)
print("Training set size:", len(x_train))
print("Testing set size:", len(x_test))

print()

# K-Fold cross validation
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

kfold = KFold(n_splits=5, shuffle=True, random_state=5)
scores = cross_val_score(model, x, y, cv = kfold)
print("Cross validation scores:", scores)
print("Average mean scores:", scores.mean())
