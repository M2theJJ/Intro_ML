# the train.csv file needs to be in the same directory as the main file

# importing modules
import matplotlib
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('tkagg')
# importing the data from the csv file
file = np.genfromtxt('train.csv', delimiter=',')

# file[x:y, s:t] -> in rows x to y take elements s to t
y = file[1:, 0]      # Ignoring the header line, the first column is the y array
X = file[1:, 1:]     # Ignoring the header line, the rest is the X array

# Only use feature x5, and x7. Maybe x2, x4, x6, and x12
# i. e. skip x0, x1, x3, x8, x9, x10, x11
# X1 = X_tot[:, 5]



# for i in range(13):
#     print(i+1)
#     plt.plot(X[:,i],y,'bo')
#     plt.show()
# lambdas array
lambdas = np.array([0.1, 1, 10, 100, 200])


# result array
res = np.zeros(len(lambdas))


# calculating the RSME for every given hyper parameter in lmb
def rsme(lm):
    # define the regression model
    model = linear_model.Ridge(alpha=lm)

    # TODO: should we take cross_val_predict or cross_val_score?
    # use 10-fold CV to evaluate model
    predictions = model_selection.cross_val_predict(model, X[:, 5:13], y, cv=10)

    # print(predictions)
    # print(len(predictions))
    # TODO: are we calculating the RSME correctly?
    return np.sqrt(metrics.mean_squared_error(y, predictions))


# calculate the RSME
for i in range(len(lambdas)):
    res[i] = rsme(lambdas[i])
    # print(res[i])


# export the results in the csv file
# TODO: rename the output file
np.savetxt('res.csv', res, delimiter=',')
