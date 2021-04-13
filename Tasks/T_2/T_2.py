# Some general information:
# We got 18995 patients.
# Each patient has 12 measurements, thus in total we got 227940 measurements (test_features.csv)


# We could try to impute the missing data with the iterative imputer class from scikit-learn:
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
# I got inspired by this website when it came to the imputation for our dataset
# https://machinelearningmastery.com/iterative-imputation-for-missing-values-in-machine-learning/

### This code gives a metric on how many data points are missing in each feature
from pandas import read_csv
dataframe = read_csv('train_features.csv', header=0)

for i in range(dataframe.shape[1]):
	n_miss = dataframe.iloc[:, i].isnull().sum()
	perc = n_miss / dataframe.shape[0] * 100
	print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))


### This code automatically imputes all the data, but it does not give satisfying results
### i.e. the only measured value for 'AST' is 20 and the eleven imputed data points arcan easily be a magnitude bigger.
### For me, this does not make sense as I would expect the other values to be around that measured value, too.
# import numpy as np
# import pandas as pd
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
#
# # Load dataset
# dataframe = pd.read_csv('train_features.csv', header=0)
#
# # Split into input and output elements
# data = dataframe.values
# ix = [i for i in range(data.shape[1])]
# X, y = data[:, ix], data[:, data.shape[1]-1]
#
# # Define imputer and fit on dataset
# imputer = IterativeImputer(max_iter=5).fit(X)
#
# # The dataset Xtrans contains no 'nan' values anymore
# Xtrans = imputer.transform(X)
#
# # Export the results in the csv file
# np.savetxt('train_features_transformed.csv', Xtrans, delimiter=',', fmt='%1.2f')



# Sub-task 1: Ordering of medical test
# Here we are interested in anticipating the future needs of the patient. You have to predict whether a certain medical
# test is ordered by a clinician in the remaining stay. This sub-task is a binary classification : 0 means that there
# will be no further tests of this kind ordered whereas 1 means that at least one is ordered in the remaining stay.
#
# The corresponding columns containing the binary ground truth in train_labels.csv are: LABEL_BaseExcess,
# LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2,
# LABEL_Bilirubin_direct, LABEL_EtCO2.
#
# Because there is an imbalance between labels in these sub-tasks we evaluate the performance of a model with the
# Area Under the Receiver Operating Characteristic Curve (ROC Curve), which is a threshold-based metric. To achieve
# good performance, it is important to produce (probabilistic) real-valued predictions in the interval [0, 1].
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score


# Sub-task 2: Sepsis prediction
#
# In this sub-task, we are interested in anticipating future life-threatening events. You have to predict whether a
# patient is likely to have a sepsis event in the remaining stay. This task is also a binary classification : 0 means
# that no sepsis will occur, 1 otherwise.
#
# The corresponding column containing the binary ground-truth in train_labels.csv is LABEL_Sepsis.
#
# This task is also imbalanced, thus we’ll also evaluate performance using Area Under the ROC Curve.
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score


# Sub-task 3: Keys vitals signs prediction
#
# In this type of sub-task, we are interested in predicting a more general evolution of the patient state. To this
# effect, here we aim at predicting the mean value of a vital sign in the remaining stay. This is a regression task.
#
# The corresponding columns containing the real-valued ground truth in train_labels.csv are: LABEL_RRate, LABEL_ABPm,
# LABEL_SpO2, LABEL_Heartrate.
#
# To evaluate the performance of a given model on this sub-task we use R^2 Score.
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html


# This code writes a pandas dataframe directly into a zip file, supposed df is a pandas dataframe containing the result
# df.to_csv('result.zip', float_format='%.3f', header=True, index=False, compression='zip')
