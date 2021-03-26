# the train.csv file needs to be in the same directory as the main file

# importing modules
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import numpy as np


# importing the data from the csv file
file = np.genfromtxt('train.csv', delimiter=',')

# file[x:y, s:t] -> in rows x to y take elements s to t
y = file[1:, 0]      # Ignoring the header line, the first column is the y array
X = file[1:, 1:]     # Ignoring the header line, the rest is the X array


# lambdas array
lmb = np.array([[0.1], [1], [10], [100], [200]])

# result array
res = np.zeros(len(lmb))

for i in range(5):
    # define the regression model
    model = linear_model.RidgeCV(alphas=lmb[i], cv = 10)

    # compute the best coefficients based on model
    regr = model.fit(X,y)

    # get the yhats
    predictions = model.predict(X)
    #print(predictions)

    # print(len(predictions))
    res[i] = metrics.mean_squared_error(y, predictions, squared = False)


# export the results in the csv file
print(res)
np.savetxt('res.csv', res, delimiter=',')