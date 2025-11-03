# Numpy Library

# Creating an array (1D)
import numpy as np
arr = np.array([1,2,3])
print("1D array is:", arr)

print()

# Creating an array (2D)
arr1 = np.array([[1,2,3], [4,5,6]])
print("2D ARRAY IS :",arr1)

print()

# Creating an array from tuple
arr2 = np.array((1,3,6))
print(arr2)

print()

# Accessing the array index
arr3 = np.array([[-1, 2, 0, 4],
                [4, -0.5, 6, 0],
                [2.6, 0, 7, 8],
                [3, -7, 4, 2.0]])

a = arr3[:2, ::2] # :2 denotes a slice that includes the first two rows (index 0 and 1).
# ::2 denotes a slice that includes columns with a step of 2 starting from the first column (i.e., every other column).

print(a)

print()

print("Basic Array operations")

# Basic Array operations
b = np.array([[1,2], [3,4]])
c = np.array([[5,6], [7,8]])

print("Adding 1 to every element", b + 1)
print("Subtracting 2 from every element", c - 2)
print("Adding two arrays", b + c)

print()

# Constructing a datatype object
print("Constructing a datatype object")
x = np.array([1,2,3])
print(x.dtype)

print()

y = np.array([1.2, 3.4])
print(y.dtype)

print()

# Square root of any array
arr4 = np.array([1,2])
sqrt = np.sqrt(arr4)
print("Square root of an array:", sqrt)

print()

# Transpose of an array
arr5 = np.array([[5,6], [7,8]])
Trans_arr = arr5.T
print("Transpose of an array:", Trans_arr)

print()

# Sum of two lists (Here it is concatenation only)
d = [1,2,3,4,5]
e = [6,7,8,9,10]
f = d + e
print(f)

print()

# Sum of two arrays (Here it means calculation)
g = np.array([1,2,3,4,5])
h = np.array([6,7,8,9,0])
output = g + h
print(output)

print()

# Another example of sum
i = [100]
output1 = g + i
print(output1)

print()

# 2D list (list of list)
print("2D List")
j = np.array([[1,2,3,4,5,6,7,8,9,10]])
print(j)

print()

# 2D array with 3 elements
print("2D Array with 3 elements")
k = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20], [10,9,8,7,6,5,4,3,2,1]])
print(k)

print()

# 3D array
print("3D Array")
l = np.array([[[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]]])
print(l)

print()
print("-----------------MULTIDIMENSIONAL ARRAY---------------------")
# Multidimensional array
m = np.array([[[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]],
             [[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]]])
print(m)

print()

# No. of brackets tells about dimensions
print(len(j))
print(len(k))
print(len(l))
print(len(m))

print()

print(j.shape)
print(k.shape)
print(l.shape)
print(m.shape)

print()

print(j.ndim)
print(k.ndim)
print(l.ndim)
print(m.ndim)

print()

# dtype gives the type
print(j.dtype)
print(l.dtype)

print()

print("----------------- Slicing of an array ------------------")
np.random.seed(2)
x = np.random.randint(1,100,(10,10))
print(x)

print()

print(x[:,:])

print()

print(x[:, 4:8])

print()

print(x[1,4])

print()

print(x[3:6, :])

print()

print(x[6:,:])

print()

print(x[1][4])

print("------------------------------Array Functions/Methods-----------------------")
np.random.seed(2)
x = np.random.randint(1,100,(5,6))
print(x)

print()
print("-------SUM-----")
print("Sum: ",x.sum())
print("Column Wise Sum: ", x.sum(axis=0))
print("Row Wise Sum: ", x.sum(axis=1))

print()

print("------Mean------")
print("Mean:", x.mean())
print("Column wise mean: ", x.mean(axis=0))
print("Row wise mean:", x.mean(axis=1))

print()

print("------Min & Max------")
print("Minimum:", x.min())
print("Minimum at X axis", x.min(axis=0))
print("Minimum at Y axis", x.min(axis=1))
print("Maximum:", x.max())
print("Maximum at X axis", x.max(axis=0))
print("Maximum at Y axis", x.max(axis=1))

print("ArgMax", x.argmax())
print("x axis argmax", x.argmax(axis=0)) # Maximum indices along columns
print("x axis argmax", x.argmax(axis=1)) # Maximum indices along rows

print()

print("--------------Using astype--------------")
x_1 = x.astype('float')
print("x_1 is :",x_1)

print()

x_2 = x.astype('complex')
print("x_2 is :",x_2)

print()

print("Shape of x:", x.shape)

print("-------------------RESHAPING--------------")
print("Reshaping of x:", x.reshape(30))
print("Reshaping of x with different value:", x.reshape((10,3)))
print("Another reshaping:", x.reshape([5,2,3]))

print()

# Creating an array
n = np.array([1.4689,22,33.4,7,10,23.4444,90])
print("An Array:", n)

print()

print("Rounding off:", np.round(n))

print("Rounding off by two decimals:", np.round(n,2))

print()

n1 = np.array([45,32,323,22,22,22,45,45,333,323,222])
print(n1)

print("Unique values from n1:", np.unique(n1))

print()

print(np.unique(n1, return_counts=True))

# Random numbers
print("------------------RANDOM NUMBERS---------------------")
print(np.random.randint(100))

print(np.random.seed(100))

print("----------------`")

print(np.random.randint(100,500,(3,4,2)))

print("-----------------")

print(np.random.randint(100,500,10))

print("------------------")

print(np.random.uniform(100,500,10))

print()

print(np.zeros([4,5]))

print()

print(np.ones([4,5]))

print()

o = np.random.randint(0, 30, 80)
print("Some random numbers :", o)

print()

print("Variance of random numbers :", o.var())  # variance

print()

print("Standard deviation of random numbers:", o.std()) # standard deviation

print()

print(np.arange(5,25,2))

print()

print(np.arange(4, 25, 4).reshape(2, 3))

print()

p = np.random.randint(1,11,(2,5))
print("Value of p :", p)

print()

q = np.random.randint(1,18,(2,5))
print("Value of q :", q)

print()

print("Addition of p and q is :",np.add(p,q))

print("Subtraction of p and q is:", np.subtract(p,q))

print()

r = np.random.randint(1,11,(5,2))
print(r)

print()

print("Multiplication of q and r is :", np.matmul(q,r))