import numpy as np
import pandas as pd
from pandas import read_csv

import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler   
from sklearn.model_selection import train_test_split
from sklearn import datasets

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

# Data: patient over time for each particle
# For subtask 1
baseExcessTab = colToMatrixNoSup(train_features, 10)
refBaseExcess = ground_truth[ :,1]
fibrinogeneTab = colToMatrixNoSup(train_features, 12)
refFibri = ground_truth[ :,2]
astTab = colToMatrixNoSup(train_features, 17)
refAst = ground_truth[ :,3]
alkTab = colToMatrixNoSup(train_features, 27)
refAlk = ground_truth[ :,4]
bilTotTab = colToMatrixNoSup(train_features, 33)
refBilTot = ground_truth[ :,5]
lactTab= colToMatrixNoSup(train_features, 6)
refLact = ground_truth[ :,6]
tropoTab = colToMatrixNoSup(train_features, 34)
refTropo = ground_truth[ :,7]
sao2Tab = colToMatrixNoSup(train_features, 20)
refSao2 = ground_truth[ :,8]
bilDirTab = colToMatrixNoSup(train_features, 29)
refBilDir = ground_truth[ :,9]
etco2Tab = colToMatrixNoSup(train_features, 3)
refEtco2 = ground_truth[ :,10]

# For the final output
# For subtask 1
baseExcessTest = colToMatrixNoSup(test_features, 10)
fibrinogeneTest = colToMatrixNoSup(test_features, 12)
astTest = colToMatrixNoSup(test_features, 17)
alkTest = colToMatrixNoSup(test_features, 27)
bilTotTest = colToMatrixNoSup(test_features, 33)
lactTest= colToMatrixNoSup(test_features, 6)
tropoTest = colToMatrixNoSup(test_features, 34)
sao2Test = colToMatrixNoSup(test_features, 20)
bilDirTest = colToMatrixNoSup(test_features, 29)
etco2Test = colToMatrixNoSup(test_features, 3)

n_patient = baseExcessTest.shape[0]
act_col = 10
cols = [10, 12, 17, 27, 33, 6, 34, 20, 29, 3]
meanAcc = 0

res = torch.zeros(n_patient, act_col+1)

# Automatize a bit
Xst =np.array([baseExcessTest, fibrinogeneTest, astTest, alkTest, bilTotTest, lactTest, tropoTest, sao2Test, bilDirTest, etco2Test])
Xs = np.array([baseExcessTab, fibrinogeneTab, astTab, alkTab, bilTotTab, lactTab, tropoTab, sao2Tab, bilDirTab, etco2Tab])
ys = np.array([refBaseExcess, refFibri, refAst, refAlk, refBilTot, refLact, refTropo, refSao2, refBilDir, refEtco2])
Xs_train = act_col*[[[]]]
Xs_test = act_col*[[]]
ys_train = act_col*[[]]
ys_test = act_col*[[]]

# Takes one third of the input for validation
for i in range(act_col):
    Xs_train[i], Xs_test[i], ys_train[i], ys_test[i] = train_test_split(Xs[i], ys[i], test_size=0.20, random_state=69)

# Standardize
for i in range(act_col):
    scaler = StandardScaler()
    Xs_train[i] = scaler.fit_transform(Xs_train[i])
    Xs_test[i] = scaler.transform(Xs_test[i])
    Xst[i] = scaler.fit_transform(Xst[i])
    
# Conversion to Tensor
Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
ys_train = torch.tensor(ys_train, dtype=torch.float32)
Xs_test = torch.tensor(Xs_test, dtype=torch.float32)
ys_test = torch.tensor(ys_test, dtype=torch.float32)
Xst = torch.tensor(Xst, dtype=torch.float32)

class LogReg(nn.Module):
    def __init__(self, n_input_features):
        super(LogReg,self).__init__()
        self.linear1 = nn.Linear(n_input_features, 50)
        self.linear2 = nn.Linear(50, 25)
        self.linear3 = nn.Linear(25, 1)

    def forward(self, x):
        y_predicted = torch.tanh(self.linear1(x))
        y_predicted = torch.tanh(self.linear2(y_predicted))
        y_predicted = torch.sigmoid(self.linear3(y_predicted))
        self.probabilities = y_predicted
        y_predicted = torch.sigmoid(self.probabilities)
        return y_predicted
# Get size
n_samples, n_features = Xs_train[j].shape

# Define Model
input_size = n_features
output_size = 1
model = LogReg(input_size)

#Training
n_iters = 100
learning_rate = 0.01

# loss = MSE
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for j in range(act_col):
    
    for epoch in range(n_iters):
        # Predict
        # y_pred = forward(Xs_train[0])
        y_pred = model(Xs_train[j])

        # Compute loss
        loss = criterion(y_pred.squeeze(), ys_train[j])

        # Compute gradients
        loss.backward() #dloss/dx

        # Update weight
        optimizer.step()

        # We need to reinitialize weight not to accumulate
        optimizer.zero_grad()
        
        #if epoch % 10 == 0:
        #    print(loss)

    with torch.no_grad():
        y_predicted = model(Xst[j]).squeeze()
        res[:,j] = model.probabilities.squeeze()
        #y_predictedd = model(Xs_test[j]).squeeze()
        #y_predictedd_cls = y_predictedd.round()
        #acc = y_predictedd_cls.eq(ys_test[j]).sum()/float(ys_test[j].shape[0])
        #print(acc)
        #meanAcc = meanAcc + acc

###########################################################################################################################

# For subtask 2
tempTab = colToMatrixNoSup(train_features, 7)
heartTab = colToMatrixNoSup(train_features, 32)
rrateTab = colToMatrixNoSup(train_features, 11)
wbcTab = colToMatrixNoSup(train_features, 14)
refSepsis = ground_truth[:,11]

n_patient = tempTab.shape[0]

sepTrain = np.zeros((n_patient, 4*n_features))
for i in range(n_patient):
    sepTrain[i] = np.append(tempTab[i], [heartTab[i], rrateTab[i], wbcTab[i]])

tempTest = colToMatrixNoSup(test_features, 7)
heartTest = colToMatrixNoSup(test_features, 32)
rrateTest = colToMatrixNoSup(test_features, 11)
wbcTest = colToMatrixNoSup(test_features, 14)

n_test_patient = tempTest.shape[0]
sepTest = np.zeros((n_test_patient, 4*n_features))
for i in range(n_test_patient):
    sepTest[i] = np.append(tempTest[i], [heartTest[i], rrateTest[i], wbcTest[i]])
    
X_train, X_test, y_train, y_test = train_test_split(sepTrain, refSepsis, test_size=0.20, random_state=69)

n_sample, n_features = X_train.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sepTest = scaler.fit_transform(sepTest)

# Conversion to Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
sepTest = torch.tensor(sepTest, dtype=torch.float32)

class LogRegT2(nn.Module):
    def __init__(self, n_input_features):
        super(LogRegT2,self).__init__()
        self.linear1 = nn.Linear(n_features, 200)
        self.linear2 = nn.Linear(200, 100)
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 25)
        self.linear5 = nn.Linear(25,1)

    def forward(self, x):
        y_predicted = torch.tanh(self.linear1(x))
        y_predicted = torch.tanh(self.linear2(y_predicted))
        y_predicted = torch.tanh(self.linear3(y_predicted))
        y_predicted = torch.tanh(self.linear4(y_predicted))
        y_predicted = torch.sigmoid(self.linear5(y_predicted))
        self.probabilities = y_predicted
        y_predicted = torch.sigmoid(self.probabilities)
        return y_predicted

# Define Model
input_size = n_features
output_size = 1
model = LogRegT2(input_size)

#Training
n_iters = 1000
learning_rate = 0.005

# loss = MSE
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # Predict
    y_pred = model(X_train)

    # Compute loss
    loss = criterion(y_pred.squeeze(), y_train)

    # Compute gradients
    loss.backward() #dloss/dx

    # Update weight
    optimizer.step()

    # We need to reinitialize weight not to accumulate
    optimizer.zero_grad()
        
    if epoch % 10 == 0:
        print(loss)

    with torch.no_grad():
        y_predicted = model(sepTest).squeeze()
        res[:,act_col] = model.probabilities.squeeze()
        #y_predictedd = model(Xs_test[j]).squeeze()
        #y_predictedd_cls = y_predictedd.round()
        #acc = y_predictedd_cls.eq(ys_test[j]).sum()/float(ys_test[j].shape[0])
        #print(acc)
        #meanAcc = meanAcc + acc

#print("Average: ", meanAcc/act_col)
np.savetxt('res_2A.csv', res, delimiter=',')
