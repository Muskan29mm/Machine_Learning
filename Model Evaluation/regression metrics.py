from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("weather.csv")
print(df)

X, Y = df.iloc[:, 2].values.reshape(-1, 1), df.iloc[:, 3].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

regression = LinearRegression()
regression.fit(X_train, Y_train)
Y_pred = regression.predict(X_test)

print()

mae = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error:", mae)

print()

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

rmse = root_mean_squared_error(Y_test, Y_pred)
print("Root Mean Squared Error:", rmse)

# Mean absolute Percentage Error
mape = mean_absolute_percentage_error(Y_test, Y_pred)
print("Means Absolute Percentage Error:", mape)


