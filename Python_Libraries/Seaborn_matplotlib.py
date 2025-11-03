# Seaborn and Matplotlib

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import load_dataset

# Univariate - One variable
# Bivariate - Two variable
# Multivariate - Multiple variable

# --------------Univariate----------------------------
# Histogram
np.random.seed(1)
data = np.random.randint(1,100,50)
plt.hist(data)
plt.show()

print()

# BoxPlot
data = [1,30,34,50,45,32,34,40,45,45,34,38,30,34,50,45,32,34,40,45,45,
        34,34,100, 50,45,32,34,40,45,45,34,38,30,34,50,45,32,34,40,45,45,34,38,
        30,34,50,45,32,34,40,45,45,34,38,30,34,50,45,32,34,40,45,45,34,38,
        30,34,50,45,32,34,40,45,45,34,38,30,34,50,45,32,34,40,45,45,34,38]
sns.boxplot(y=data)
plt.show()

# ----------------Bivariate--------------------------------
# LinePlot
x = [1,3,5,7,9,11]
y = [10,20,30,40,50,70]
plt.plot(x,y)
plt.show()

x = [1,3,4,5,7,8,9]
y = [10,30,40,20,50,60,90]
sns.lineplot(x=x, y=y)
plt.show()

# ScatterPlot
x = [1,2,4,6,8,0]
y = [10,20,30,40,50,70]
plt.scatter(x,y)
plt.show()

x = [1,3,4,5,7,8,9]
y = [10,30,40,20,50,60,90]
sns.scatterplot(x=x,y=y)
plt.show()

# Bar Plot
x = [1,3,4,5,7,8,9]
y = [10,30,40,20,50,60,90]
plt.bar(x,y)
plt.show()

x = [1,3,4,5,7,8,9]
y = [10,30,40,20,50,60,90]
sns.barplot(x=x,y=y)
plt.show()

x = [1,3,4,5,7,8,9]
y = [10,30,40,20,50,60,90]
plt.scatter(x,y)
plt.plot(x,y)
plt.show()



x = [1,3,4,5,7,8,9]
y = [10,30,40,20,50,60,90]
sns.pointplot(x=x,y=y)
plt.show()

print(sns.get_dataset_names())

tips = sns.load_dataset('tips')
print(tips)

type(tips)

# Pie Chart
y = np.array([35,25,10,15])
plt.pie(y)
plt.show()

