import numpy as np
import pandas as pd
from pandas import read_csv

import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model


import matplotlib.pyplot as plt


# For each patient, we get part of their current state for 12 consecutive hours
test_features_frame = read_csv('test_features.csv', header=0)
train_features_frame = read_csv('train_features.csv', header=0)
ground_truth_frame = read_csv('train_labels.csv', header=0)

test_features = test_features_frame.values
train_features = train_features_frame.values
ground_truth = ground_truth_frame.values
res = np.zeros((round(len(test_features)/12),12))

# Data Imputer on the training data
imputer = SimpleImputer(strategy='mean')
imputer.fit(train_features_frame)
train_features = imputer.transform(train_features_frame)
imputer.fit(test_features_frame)
test_features = imputer.transform(test_features_frame)

def colToMatrixNoSup(array, colId):
    notnan = 0
    table = []
    temp = []
    for i in range(round(len(array)/12)):
        for j in range(12):
            temp.append(array[i*12+j][colId])
        table.append(temp)
        temp = []
    return np.asarray(table)

def twelveToStat(array):
    temp1 = np.mean(array)
    temp2 = np.max(array)
    temp3 = np.min(array)
    temp4 = np.median(array)
    temp5 = np.var(array)
    return [temp1, temp2, temp3, temp4, temp5]

# Data: patient over time for each particle
# # For subtask 1
baseExcessTab = colToMatrixNoSup(train_features, 10)
# refBaseExcess = ground_truth[ :,1]
# fibrinogeneTab = colToMatrixNoSup(train_features, 12)
# refFibri = ground_truth[ :,2]
# astTab = colToMatrixNoSup(train_features, 17)
# refAst = ground_truth[ :,3]
# alkTab = colToMatrixNoSup(train_features, 27)
# refAlk = ground_truth[ :,4]
# bilTotTab = colToMatrixNoSup(train_features, 33)
# refBilTot = ground_truth[ :,5]
# lactTab= colToMatrixNoSup(train_features, 6)
# refLact = ground_truth[ :,6]
# tropoTab = colToMatrixNoSup(train_features, 34)
# refTropo = ground_truth[ :,7]
# sao2Tab = colToMatrixNoSup(train_features, 20)
# refSao2 = ground_truth[ :,8]
# bilDirTab = colToMatrixNoSup(train_features, 29)
# refBilDir = ground_truth[ :,9]
# etco2Tab = colToMatrixNoSup(train_features, 3)
# refEtco2 = ground_truth[ :,10]
#
# act_col = 10
# cols = [10, 12, 17, 27, 33, 6, 34, 20, 29, 3]
n_patient = baseExcessTab.shape[0]
#
# # For the final output
# # For subtask 1
baseExcessTest = colToMatrixNoSup(test_features, 10)
# fibrinogeneTest = colToMatrixNoSup(test_features, 12)
# astTest = colToMatrixNoSup(test_features, 17)
# alkTest = colToMatrixNoSup(test_features, 27)
# bilTotTest = colToMatrixNoSup(test_features, 33)
# lactTest= colToMatrixNoSup(test_features, 6)
# tropoTest = colToMatrixNoSup(test_features, 34)
# sao2Test = colToMatrixNoSup(test_features, 20)
# bilDirTest = colToMatrixNoSup(test_features, 29)
# etco2Test = colToMatrixNoSup(test_features, 3)
#
n_patient_test = baseExcessTest.shape[0]
#
# meanAcc = 0
# res = np.zeros((n_patient_test, 15))
#
# # Automatize a bit
# Xst =np.array([baseExcessTest, fibrinogeneTest, astTest, alkTest, bilTotTest, lactTest, tropoTest, sao2Test, bilDirTest, etco2Test])
# Xs = np.array([baseExcessTab, fibrinogeneTab, astTab, alkTab, bilTotTab, lactTab, tropoTab, sao2Tab, bilDirTab, etco2Tab])
# ys = np.array([refBaseExcess, refFibri, refAst, refAlk, refBilTot, refLact, refTropo, refSao2, refBilDir, refEtco2])
#
# Xs_train = act_col*[[[]]]
# Xs_test = act_col*[[]]
# ys_train = act_col*[[]]
# ys_test = act_col*[[]]
#
# # Takes one third of the input for validation
# for i in range(act_col):
#     Xs_train[i], Xs_test[i], ys_train[i], ys_test[i] = train_test_split(Xs[i], ys[i], test_size=0.01, random_state=69)
#
# # Standardize
# for i in range(act_col):
#     scaler = StandardScaler()
#     Xs_train[i] = scaler.fit_transform(Xs_train[i])
#     Xs_test[i] = scaler.transform(Xs_test[i])
#     Xst[i] = scaler.fit_transform(Xst[i])
#
# print("Transforming features")
#
# Xs_trainn = np.zeros((act_col, len(Xs_train[0]), 5))
# Xs_testt = np.zeros((act_col, len(Xs_test[0]), 5))
# Xstt = np.zeros((act_col, len(Xst[0]), 5))
# # Extract stats:
# for i in range(act_col):
#     for j in range(int(n_patient*0.99)):
#         Xs_trainn[i][j] = twelveToStat(Xs_train[i][j])
#     for j in range(int(n_patient*0.01)):
#         Xs_testt[i][j] = twelveToStat(Xs_test[i][j])
#     for j in range(n_patient_test):
#         Xstt[i][j] = twelveToStat(Xst[i][j])
#
# print("Evaluating model")
#
# model = linear_model.LogisticRegressionCV(tol=1e-07, penalty="l2", random_state = 42, max_iter= 100)
# for j in range(act_col):
#     logregr = model.fit(Xs_trainn[j], ys_train[j])
#     predictions = logregr.predict_proba(Xstt[j])
#     res[:,j] = predictions[:,0]
#
# print("Subtask 1 done")
#
# np.savetxt('res_2A.csv', res, delimiter=',', fmt='%.3f')

###########################################################################################################################

# For subtask 2
tempTab = colToMatrixNoSup(train_features, 7)
heartTab = colToMatrixNoSup(train_features, 32)
rrateTab = colToMatrixNoSup(train_features, 11)
wbcTab = colToMatrixNoSup(train_features, 14)
refSepsis = ground_truth[:,11]

tempTabb = np.zeros((n_patient, 5))
heartTabb = np.zeros((n_patient, 5))
rrateTabb = np.zeros((n_patient, 5))
wbcTabb = np.zeros((n_patient, 5))

n_patient, n_features = tempTabb.shape

tempTest = colToMatrixNoSup(test_features, 7)
n_patient_test, n_features_test = tempTest.shape
heartTest = colToMatrixNoSup(test_features, 32)
rrateTest = colToMatrixNoSup(test_features, 11)
wbcTest = colToMatrixNoSup(test_features, 14)

tempTestt = np.zeros((n_patient, 5))
heartTestt = np.zeros((n_patient, 5))
rrateTestt = np.zeros((n_patient, 5))
wbcTestt = np.zeros((n_patient, 5))

print("Transforming features")

# Extract stats:
for j in range(n_patient):
    tempTabb = twelveToStat(tempTab)
    heartTabb = twelveToStat(heartTab)
    rrateTabb = twelveToStat(rrateTab)
    wbcTabb = twelveToStat(wbcTab)
for j in range(n_patient_test):
    tempTestt = twelveToStat(tempTestt)
    heartTestt = twelveToStat(heartTestt)
    rrateTestt = twelveToStat(rrateTestt)
    wbcTestt = twelveToStat(wbcTestt)

sepTrain = np.zeros((n_patient, 4*n_features))
for i in range(n_patient):
    sepTrain[i] = np.append(tempTabb[i], [heartTabb[i], rrateTabb[i], wbcTabb[i]])

n_test_patient = tempTestt.shape[0]

sepTest = np.zeros((n_test_patient, 4*n_features))
for i in range(n_test_patient):
    sepTest[i] = np.append(tempTestt[i], [heartTestt[i], rrateTestt[i], wbcTestt[i]])

X_train, X_test, y_train, y_test = train_test_split(sepTrain, refSepsis, test_size=0.01, random_state=69)

n_sample, n_features = X_train.shape

#Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sepTest = scaler.fit_transform(sepTest)

print("Evaluating model")

model = linear_model.LogisticRegressionCV(tol=1e-07, penalty="l2", random_state = 42, max_iter= 100)
logregr = model.fit(X_train, y_train)
predictions = logregr.predict_proba(sepTest)
res[:,10] = predictions[0]

print("Task 2 finished")

np.savetxt('res_2A.csv', res, delimiter=',', fmt='%.3f')

#######################################################################################################################
# Sub Task 3

RRateTab = colToMatrixNoSup(train_features, 11)
refRRate = ground_truth[ :,12]
RRateTest = colToMatrixNoSup(test_features, 11)

ABPmTab = colToMatrixNoSup(train_features, 22)
refABPm = ground_truth[ :,13]
ABPmTest = colToMatrixNoSup(test_features, 22)

SPO2Tab = colToMatrixNoSup(train_features, 28)
refSPO2 = ground_truth[ :,14]
SP02Test = colToMatrixNoSup(test_features, 28)

HeartrateTab = colToMatrixNoSup(train_features, 32)
refHeartrate = ground_truth[ :,15]
HeartrateTest = colToMatrixNoSup(test_features, 32)

RRateTabb = np.zeros((n_patient, 5))
ABPmTabb = np.zeros((n_patient, 5))
SPO2Tabb = np.zeros((n_patient, 5))
HeartrateTabb = np.zeros((n_patient, 5))

RRateTestt = np.zeros((n_patient_test, 5))
ABPmTestt = np.zeros((n_patient_test, 5))
SPO2Testt = np.zeros((n_patient_test, 5))
HeartrateTestt = np.zeros((n_patient_test, 5))

print("Transforming features")

# Extract stats:
for j in range(n_patient):
    RRateTabb = twelveToStat(RRateTab)
    ABPmTabb = twelveToStat(ABPmTab)
    SPO2Tabb = twelveToStat(SPO2Tab)
    HeartrateTabb = twelveToStat(HeartrateTab)
for j in range(n_patient_test):
    RRateTestt = twelveToStat(RRateTest)
    ABPmTestt = twelveToStat(ABPmTest)
    SP02Testt = twelveToStat(SP02Test)
    HeartrateTestt = twelveToStat(HeartrateTest)

# Automatize a bit
Xst_3 = np.array([RRateTestt, ABPmTestt, SP02Testt, HeartrateTestt])
Xs_3 = np.array([RRateTabb, ABPmTabb, SPO2Tabb, HeartrateTabb])
ys_3 = np.array([refRRate, refABPm, refSPO2, refHeartrate])
Xs_train3 = 4*[[[]]]
Xs_test3 = 4*[[]]
ys_train3 = 4*[[]]
ys_test3 = 4*[[]]

# Takes one third of the input for validation
for i in range(4):
    Xs_train3[i], Xs_test3[i], ys_train3[i], ys_test3[i] = train_test_split(Xs_3[i], ys_3[i], test_size=0.01, random_state=69)

print("Evaluating model")

model = linear_model.LassoCV(tol=1e-07, penalty="l2", random_state = 42, max_iter= 100000)
for j in range(4):
    logregr = model.fit(Xs_train3[j], ys_train3[j])
    predictions = logregr.predict_proba(Xst_3[j])
    res[:,10+j] = predictions[:,0]

np.savetxt('res_2A.csv', res, delimiter=',', fmt='%.3f')