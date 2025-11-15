# Binning Transforms continuous numerical data into discrete intervals or "bins".

import pandas as pd
data = {'Age': [23,67,39,80,90,10,30,89,15,48,50,76]}
df = pd.DataFrame(data)

bins = [0,20,40,60,80,100]
labels = ['Teenager','Young Adult','Adult','Senior','Elderly']

df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
print(df)