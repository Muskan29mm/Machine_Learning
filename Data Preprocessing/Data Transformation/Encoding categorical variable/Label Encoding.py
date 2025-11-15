# Method 1: Label Encoding using Scikit-Learn

from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.DataFrame({
    'Fruit' : ['Mango', 'Banana', 'Cherry', 'Apple', 'Grapes'],
    'Price': [1.2, 3.0, 4.5, 2.3, 4.6]
})

# Initialize and fit Label encoder
le = LabelEncoder()
data['Fruit_Encoded'] = le.fit_transform(data['Fruit'])

print(data)
print("Category Mapping:", le.classes_)


# Method 2: Encode using pandas categorical type
data['Fruit_Encoded_Pandas'] = data['Fruit'].astype('category').cat.codes
print(data)
print("Category Mapping:", dict(enumerate(data['Fruit'].astype('category').cat.categories)))


# Method 3: Encoding Ordinal Data
data = pd.DataFrame({
    'Satisfaction': ['Low', 'High', 'Medium', 'Low', 'High'],
    'Score': [3, 8, 5, 2, 9]
})

satisfaction_order = {'Low': 0, 'Medium': 1, 'High': 2}
data['Satisfaction_Encoded'] = data['Satisfaction'].map(satisfaction_order)

print(data)