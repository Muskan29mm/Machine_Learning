# Pandas Library

# Import Pandas
import pandas as pd
import numpy as np

# Creating a series using the Pandas Library
ser = pd.Series()
print("Pandas Series:", ser)

data = np.array(['a', 'b', 'c', 'd', 'e'])

ser = pd.Series(data)
print("Pandas Series:\n", ser)

print()

# Creating a Data Frame using the Pandas Library
df = pd.DataFrame()
print(df)

lst = ["Hello", "How", "are", "You", "Hope", "you", "are", "fine"]

df = pd.DataFrame(lst)
print("Pandas Dataframe:\n", df)

print()

# Creating a dataframe from a List
lst = ['this', 'is', 'tutorial', 'for', 'Pandas', 'Dataframe']
df1 = pd.DataFrame(lst)
print(df1)

print()

# Creating a dataframe from dict of numpy array
data = np.array([[1,2,3], [4,5,6], [7,8,9]])
df2 = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df2)

print()

# Creating a dataframe from list of dictionaries
dict = {'name': ["Seema", "Neha", "Karan", "Tarun"],
        'degree' :["MBA", "BBA", "MCA", "BCOM"],
        'score' :[90, 80, 80, 70]}
df3 = pd.DataFrame(dict)
print(df3)
