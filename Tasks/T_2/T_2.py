import numpy as np
from pandas import read_csv
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# For each patient, we get part of their current state for 12 consecutive hours
test_features_frame = read_csv('test_features.csv', header=0)       # only for testing after training!
train_features_frame = read_csv('train_features.csv', header=0)     # use for training the model
ground_truth_frame = read_csv('train_labels.csv', header=0)

test_features = test_features_frame.values
train_features = train_features_frame.values
ground_truth = ground_truth_frame.values
res = np.zeros((round(len(test_features)/12),12))

# Data Imputer on the training data
imputer = SimpleImputer(strategy='mean')
imputer.fit(train_features_frame)
train_features = imputer.transform(train_features_frame)

# print(SimpleImputer(strategy='mean').fit(train_features_frame).statistics_)
# print(X[0])


# How many entries are missing in the original dataframe?
# for i in range(train_features.shape[1]):
#   n_miss = train_features_frame.iloc[:, i].isnull().sum()
#   percent = n_miss / train_features_frame.shape[0] * 100
#   print('> %d, Missing: %d (%.1f%%)'%(i, n_miss, percent))


marker = np.array(12*[-99])

def isFullNan(array):
    for i in range(len(array)):
        if np.isnan(array[i])==False:
            return False
    return True

def colToMatrix(array, colId):
    notnan = 0
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
                    notnan = notnan+1
                    meanTemp = meanTemp + temp[j]
            if(notnan != 0):
                meanTemp = meanTemp/notnan
            for j in range(12):
                if np.isnan(temp[j]):
                    temp[j] = meanTemp
        table.append(temp)
        temp = []
    return np.asarray(table)

def colToMatrixNoSup(array, colId):
    notnan = 0
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
                notnan = notnan+1
                meanTemp = meanTemp + temp[j]
        if(notnan != 0):
            meanTemp = meanTemp/notnan
        for j in range(12):
            if np.isnan(temp[j]):
                temp[j] = meanTemp
        table.append(temp)
        temp = []
    return np.asarray(table)

def clear(tab, ref, str):
    for i in range(len(tab)):
        if(np.all(tab[i] == marker)):
            np.delete(tab, i)
            np.delete(ref, i)
            i = i-1
            print('Cleared in '+str)


# Data: patient over time for each particle
# For subtask 1
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

# For subtask 2
tempTab = colToMatrix(train_features, 7)
heartTab = colToMatrix(train_features, 32)
rrateTab = colToMatrix(train_features, 11)
wbcTab = colToMatrix(train_features, 14)

# Patient with only NaN shouldn't participate to the classifier
clear(baseExcessTab, refBaseExcess, 'BE')
clear(fibrinogeneTab, refFibri, 'fib')
clear(astTab, refAst, 'ast')
clear(alkTab, refAlk, 'alk')
clear(bilTotTab, refBilTot, 'biltot')
clear(lactTab, refLact, 'lact')
clear(tropoTab, refTropo, 'tropo')
clear(sao2Tab, refSao2, 'sao2')
clear(bilDirTab, refBilDir, 'bidDir')
clear(etco2Tab, refEtco2, 'etco2')

# print('done cleaning')

# For the final output
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

act_col = 10
cols = [10, 12, 17, 27, 33, 6, 34, 20, 29, 3]

# Automatize a bit
Xst = np.array([baseExcessTest, fibrinogeneTest, astTest, alkTest, bilTotTest, lactTest, tropoTest, sao2Test, bilDirTest, etco2Test])
Xs = np.array([baseExcessTab, fibrinogeneTab, astTab, alkTab, bilTotTab, lactTab, tropoTab, sao2Tab, bilDirTab, etco2Tab])
ys = np.array([refBaseExcess, refFibri, refAst, refAlk, refBilTot, refLact, refTropo, refSao2, refBilDir, refEtco2])
Xs_train = 10*[[[]]]
Xs_test = 10*[[]]
ys_train = 10*[[]]
ys_test = 10*[[]]

# print(Xs.shape)     # Should be (10,18995,12) -> correct
# print(Xst.shape)    # should be (10,12664,12) -> correct

#Xst =np.array([baseExcessTest, fibrinogeneTest])
#Xs = np.array([baseExcessTab, fibrinogeneTab])
#ys = np.array([refBaseExcess, refFibri])
#Xs_train = 2*[[[]]]
#Xs_test = 2*[[]]
#ys_train = 2*[[]]
#ys_test = 2*[[]]

# Takes one third of the input for validation
for i in range(act_col):
    Xs_train[i], Xs_test[i], ys_train[i], ys_test[i] = train_test_split(Xs[i], ys[i], test_size=0.33, random_state=69)

# Standardize
for i in range(act_col):
    scaler = StandardScaler()
    Xs_train[i] = scaler.fit_transform(Xs_train[i])
    Xs_test[i] = scaler.transform(Xs_test[i])
    Xst[i] = scaler.transform(Xst[i])

# Conversion to Tensor
Xs_train = torch.tensor(Xs_train, dtype=torch.float32)
ys_train = torch.tensor(ys_train, dtype=torch.float32)
Xs_test = torch.tensor(Xs_test, dtype=torch.float32)
ys_test = torch.tensor(ys_test, dtype=torch.float32)
Xst = torch.tensor(Xst, dtype=torch.float32)

# Some sanity checks
print('\n#########')
print(f'(Test Data) Xst.shape: {Xst.shape}')
print(f'(Training Data) Xs_train.shape: {Xs_train.shape}')
print(f'(Training Data) Xs_test.shape: {Xs_test.shape}\n')
print(f'The dimensions of the training data add up: {Xs_train.shape[1] + Xs_test.shape[1] == train_features.shape[0]/12}')
print('#########\n')

meanAcc = 0
# tot_prob = 0

# Array to store the results for subtask 1
res_t1 = np.zeros(shape=(Xst.shape[1], Xst.shape[0]))
# print(res_t1.shape)

for j in range(act_col):
    # Get size
    n_samples, n_features = Xs_train[j].shape

    # Initialize weights
    # w = torch.tensor(12*[0.0]  , dtype=torch.float32, requires_grad=True)

    # Define Model
    input_size = n_features
    output_size = 1
    class LogReg(nn.Module):
        def __init__(self, n_input_features):
            super(LogReg,self).__init__()
            self.linear1 = nn.Linear(n_input_features, 100)
            self.linear2 = nn.Linear(100, 1)

        def forward(self, x):
            y = torch.sigmoid(self.linear1(x))
            y_predicted = torch.sigmoid(self.linear2(y))
            return y_predicted
    model = LogReg(input_size)

    #Training
    n_iters = 100
    learning_rate = 0.01

    # model prediction
    # def forward(X):
    #    return torch.matmul(X,w)

    # loss = MSE
    # def loss(y, y_predicted):
    #    return torch.dot(y-y_predicted,y-y_predicted)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_iters):
        # Predict
        # y_pred = forward(Xs_train[0])
        y_pred = model(Xs_train[j])

        # Compute loss
        loss = criterion(y_pred.squeeze(), ys_train[0])

        # Compute gradients
        loss.backward() #dloss/dx

        # Update weight
        # This makes sure gradient isnt part of the computational graph
        #with torch.no_grad():
        #    w -= learning_rate*w.grad
        optimizer.step()

        # We need to reinitialize weight not to accumulate
        #w.grad.zero_()
        optimizer.zero_grad()

        #if epoch % 10 == 0:
            #print("Epoch: ", epoch+1)
            #print("Loss: ", loss)

    with torch.no_grad():
        y_predicted = model(Xs_test[j])
        # print(y_predicted.shape)
        # tot_prob += y_predicted[0].item()
        y_predicted = y_predicted.squeeze()
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(ys_test[j]).sum()/float(ys_test[j].shape[0])
        # print(f'p={model(Xst[0])}, accuracy: {acc*100:.3f} %')
        # print(acc)
        meanAcc = meanAcc + acc

        # Applying the model to the test data and storing the probabilities in the result array
        feat = model(Xst[j])
        res_t1[j] = feat.cpu().detach().numpy()[0]

# print("Average: ", meanAcc/act_col)
# print(f'p={model(Xst[0]).shape}, Average accuracy: {(meanAcc/act_col)*100:.3f} %')

print(res_t1)
