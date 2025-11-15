# Implementation of Min-Max Scaling Normalization in Python

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    'Age': [18, 25, 40, 60],
    'Salary': [30000, 50000, 80000, 120000]
}

# Create a DataFrame
df = pd.DataFrame(data)
print("Original Data:\n", df)

# Initialize the Min-Max Scaler
scaler = MinMaxScaler()

# Fit and transform the data
normalized_data = scaler.fit_transform(df)

normalized_data = pd.DataFrame(normalized_data, columns=df.columns)

print("\nNormalized Data (Min-Max Scaling):\n", normalized_data)