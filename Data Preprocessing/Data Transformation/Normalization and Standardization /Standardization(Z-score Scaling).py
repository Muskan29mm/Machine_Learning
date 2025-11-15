# Implementation of Z-score Scaling Standardization in Python

import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'Age': [18, 25, 40, 60],
    'Salary': [30000, 50000, 80000, 120000]
}

# Initialize the DataFrame
df = pd.DataFrame(data)
print("Original Data: \n",df)

# Initialize the Standard Scaler
scaler = StandardScaler()

# Fit and transform the data
standardized_data = scaler.fit_transform(df)

# Convert to DataFrame
standardized_data = pd.DataFrame(standardized_data, columns=df.columns)

print("Standardized Data (Z-score Scaling):\n", standardized_data)