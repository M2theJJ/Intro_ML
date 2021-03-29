
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# importing the data from the csv file
file = np.genfromtxt('train.csv', delimiter=',')

# file[x:y, s:t] -> in rows x to y take elements s to t
y = file[1:, 1]   # Ignoring the header line, the first column is the y array
X = file[1:, 2:]  # Ignoring the header line, the rest is the X array

# Create a matrix that holds all transformed columns (takes care of the constant transformation)
X_transformed = np.ones((700, 21))

# Linear transformation
X_transformed[:, 0:5] = X

# Quadratic transformation
X_transformed[:, 5:10] = X ** 2

# Exponential transformation
X_transformed[:, 10:15] = np.e ** X

# Cosine transformation
X_transformed[:, 15:20] = np.cos(X)

# print(y.shape)
# print(X_transformed.shape)
lmb = np.array([0.1, 1, 10, 100, 200])
model = linear_model.RidgeCV(alphas=lmb, cv = 10).fit(X_transformed, y)


# export the results in the csv file
np.savetxt('res_1B.csv', model.coef_, delimiter=',')