# Implementation of a Random Forest Classifier using scikit-learn
from pyexpat import features

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
titanic_data = pd.read_csv('../titanic.csv')
print(titanic_data.head())

titanic_data.info()

titanic_data= titanic_data.dropna(subset=['Survived'])

# Split the dataset into features and target variable
features = titanic_data[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',  'Fare', 'Embarked']]
df = titanic_data[features + ['Survived']]

df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, y , test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, Y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
classification_report = classification_report(Y_test, y_pred)

print('Accuracy of Random Forest Classifier: {accuracy * 100:.2f}%')
print("Classification Report:")
print(classification_report)

sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print("Sample Passenger: {sample_dict}")
print("Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did not Survive'}")