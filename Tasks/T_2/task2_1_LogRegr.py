import numpy as np
import random as rnd
from sklearn import linear_model
from sklearn import metrics
from pandas import read_csv

# For each patient, we get part of their current state for 12 consecutive hours
test_features_frame = read_csv('test_features.csv', header=0)
train_features_frame = read_csv('train_features.csv', header=0)
ground_truth_frame = read_csv('train_labels.csv', header=0)

test_features = test_features_frame.values
train_features = train_features_frame.values
ground_truth = ground_truth_frame.values

marker = np.array(12*[-99])

def isFullNan(array):
    for i in range(len(array)):
        if np.isnan(array[i])==False:
            return False
    return True

def colToMatrix(array, colId):
    table = []
    temp = []
    for i in range(round(len(array)/12)):
        for j in range(12):
            temp.append(array[i*12+j][colId])
        if(isFullNan(temp)):
            temp = marker
        else:
            meanTemp = 0
            for j in range(12):
                if np.isnan(temp[j]):
                    continue
                else:
                    meanTemp = meanTemp + temp[j]
            meanTemp = meanTemp/12
            for j in range(12):
                if np.isnan(temp[j]):
                    temp[j] = meanTemp
        table.append(temp)
        temp = []
    return np.asarray(table)

def colToMatrixNoSup(array, colId):
    table = []
    temp = []
    for i in range(round(len(array)/12)):
        for j in range(12):
            temp.append(array[i*12+j][colId])
        meanTemp = 0
        for j in range(12):
            if np.isnan(temp[j]):
                continue
            else:
                meanTemp = meanTemp + temp[j]
        meanTemp = meanTemp/12
        for j in range(12):
            if np.isnan(temp[j]):
                temp[j] = meanTemp
        table.append(temp)
        temp = []
    return np.asarray(table)
    
def clear(tab, ref):
    for i in range(len(tab)):
        if(np.all(tab[i] == marker)):
            np.delete(tab, i)
            np.delete(ref, i)
            i = i-1

# Data: patient over time for each particle
baseExcessTab = colToMatrix(train_features, 10)
refBaseExcess = ground_truth[ :,1]
fibrinogeneTab = colToMatrix(train_features, 12)
refFibri = ground_truth[ :,2]
astTab = colToMatrix(train_features, 17)
refAst = ground_truth[ :,3]
alkTab = colToMatrix(train_features, 27)
refAlk = ground_truth[ :,4]
bilTotTab = colToMatrix(train_features, 33)
refBilTot = ground_truth[ :,5]
lactTab= colToMatrix(train_features, 6)
refLact = ground_truth[ :,6]
tropoTab = colToMatrix(train_features, 34)
refTropo = ground_truth[ :,7]
sao2Tab = colToMatrix(train_features, 20)
refSao2 = ground_truth[ :,8]
bilDirTab = colToMatrix(train_features, 29)
refBilDir = ground_truth[ :,9]
etco2Tab = colToMatrix(train_features, 3)
refEtco2 = ground_truth[ :,10]

# Patient with only NaN shouldn't participate to the classifier
clear(baseExcessTab, refBaseExcess)
clear(fibrinogeneTab, refFibri)
clear(astTab, refAst)
clear(alkTab, refAlk)
clear(bilTotTab, refBilTot)
clear(lactTab, refLact)
clear(tropoTab, refTropo)
clear(sao2Tab, refSao2)
clear(bilDirTab, refBilDir)
clear(etco2Tab, refEtco2)

# The model we use
model = linear_model.LogisticRegression(penalty='l2', tol=1e-4, random_state=1, max_iter=10000)

# Compute regression for each particle
mBE = model.fit(baseExcessTab, refBaseExcess)
mFib = model.fit(fibrinogeneTab, refFibri)
mAst = model.fit(astTab, refAst)
mAlk = model.fit(alkTab, refAlk)
mBT = model.fit(bilTotTab, refBilTot)
mLact = model.fit(lactTab, refLact)
mTropo = model.fit(tropoTab, refTropo)
mSao2 = model.fit(sao2Tab, refSao2)
mBD = model.fit(bilDirTab, refBilDir)
mEt = model.fit(etco2Tab, refEtco2)

cols = [10, 12, 17, 27, 33, 6, 34, 20, 29, 3]
models = [mBE, mFib, mAst, mAlk, mBT, mLact, mTropo, mSao2, mBD, mEt]
res = np.zeros((round(len(test_features)/12),12))
for i in range(len(cols)):
    test = colToMatrixNoSup(test_features, cols[i])
    predictions = models[i].predict(test)
    res[ :,i] = predictions

print(res)
