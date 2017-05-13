# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# make X a matrix
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)"""

# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)

# fitting decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
trap_regressor = DecisionTreeRegressor(random_state=0)
trap_regressor.fit(X, y)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(poly_reg.fit_transform(X), y)


# visualizing linear and svr regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.plot(X, trap_regressor.predict(X), color='green')
# trap
plt.plot(X_grid, trap_regressor.predict(X_grid), color='red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color='black')
plt.title('Salary vs. Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# predicting with linear regression
lin_reg.predict(6.5)
y_lin_pred = lin_reg.predict(X)
# predixting with polymonial regression
# svr.predict(sc_X.transform(np.array([[6.5]])))
#sc_y.inverse_transform(svr.predict(sc_X.transform([[6.5]])))
trap_regressor.predict(6.5)
regressor.predict(poly_reg.fit_transform(6.5))
#y_poly_pred = svr.predict(X)
