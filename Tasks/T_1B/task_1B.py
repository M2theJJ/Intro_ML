# the train_1B.csv file needs to be in the same directory as the main file

import numpy as np
from sklearn.linear_model import LinearRegression

# importing the data from the csv file
file = np.genfromtxt('train_1B.csv', delimiter=',')

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

model = LinearRegression().fit(X_transformed, y)

# export the results in the csv file
np.savetxt('res_1B.csv', model.coef_, delimiter=',')
