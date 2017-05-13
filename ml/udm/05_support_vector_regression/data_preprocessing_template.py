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
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# fitting svr regression to the dataset
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(X, y)


# visualizing linear and svr regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.plot(X, svr.predict(X), color='green')
plt.title('Salary vs. Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# predicting with linear regression
lin_reg.predict(6.5)
y_lin_pred = lin_reg.predict(X)
# predixting with polymonial regression
# svr.predict(sc_X.transform(np.array([[6.5]])))
sc_y.inverse_transform(svr.predict(sc_X.transform([[6.5]])))
y_poly_pred = svr.predict(X)
