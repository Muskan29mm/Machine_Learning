# Data Cleaning

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fontTools.subset import subset

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

# Getting Information about data
print(df.info())

# Getting Description about data
print(df.describe())

print()

# Check for duplicate rows
# duplicated function returns a boolean Series indicating duplicate rows
print(df.duplicated())


print()

# Identify columns Datatypes
cat_col = [col for col in df.columns if df[col].dtype == 'object']
num_col = [col for col in df.columns if df[col].dtype != 'object']

print("Categorical Columns are:", cat_col)
print("Numerical Columns are:", num_col)

print()

# Count unique values in categorical columns
# Syntax: df[numeric_columns].nunique(): Returns count of unique values per column.
print(df[cat_col].nunique())

print()

# Calculate missing values as Percentage
# Detects Missing values, returning boolean data frame
print((round((df.isnull().sum() / df.shape[0]) * 100, 2)))

print(df.columns)

# Drop Irrelevant columns
df1 = df.drop(columns=['Name', 'Ticket', 'Cabin'])
df1 = df1.dropna(subset=['Embarked']).copy()  # ensure copy to avoid SettingWithCopyWarning
df1['Age'] = df1['Age'].fillna(df1['Age'].mean())

print()

# Drop duplicate columns
df1.drop_duplicates(inplace=True)

print()

# Detect Outlier with boxplot
plt.boxplot(df1['Age'].dropna(), vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()

# Handle Outliers
# Outliers are detected using box plot, now to handle them
Q1 = df1['Age'].quantile(0.25)
Q3 = df1['Age'].quantile(0.75)
IQR = Q3 - Q1
df1 = df1[(df1['Age'] >= Q1 - 1.5 * IQR) & (df1['Age'] <= Q3 + 1.5 * IQR)]

# Boxplot after removing outliers
plt.boxplot(df1['Age'].dropna(), vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()