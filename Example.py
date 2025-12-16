# Example of Training the model using dataset (Performing EDA, Data Preprocessing, One-Hot Encoding, Feature Scaling, Model Training and Evaluation)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('titanic.csv')
print(df)

print(df.head())

print(df.tail())

print((df.info()))

print(df.describe())

print(df.isnull().sum())


df['Age'] = df['Age'].fillna(df['Age'].mean())
print()

#print(df.isnull().sum())

df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0])
print()

#print(df.isnull().sum())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print()

print(df.isnull().sum())

print(df)

# Select Features
X = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Embarked']]

categorical_columns = ['Embarked']

print("-------------PERFORMING ONE-HOT ENCODING-----------------")
# Encode categorical variables(Embarked)
# One Hot Encoder
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(X[categorical_columns])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))

X_encoded = pd.concat([X.drop(columns=categorical_columns), encoded_df], axis=1)

print(X_encoded.head())

# separate target and features
y = X_encoded['Survived']
x_final = X_encoded.drop(columns=['Survived'])

# split
X_train, X_test, y_train, y_test = train_test_split(x_final, y, test_size=0.5, random_state=2)


# Scale numeric columns
numeric_cols = ['Pclass', 'Age']

# converting into type 'float'
X_train[numeric_cols] = X_train[numeric_cols].astype(float)
X_test[numeric_cols] = X_test[numeric_cols].astype(float)


scaler = StandardScaler()
X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train model
cls = LogisticRegression(max_iter=1000)
cls.fit(X_train, y_train)

print()

# Evaluate
y_pred = cls.predict(X_test)
print("Accuracy:-", accuracy_score(y_test, y_pred))
print()

print("Confusion Matrix:- ", confusion_matrix(y_test, y_pred))
print()

print("Classification Report:- ", classification_report(y_test, y_pred))

