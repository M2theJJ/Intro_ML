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


# Set the time interval for every patient to 0 to 11
train_features_frame['Time'] = train_features_frame.groupby('pid')['Time'].transform(lambda x: x - x.min())


test_features = test_features_frame.values
train_features = train_features_frame.values
ground_truth = ground_truth_frame.values
res = np.zeros((round(len(test_features)/12),12))

# Data Imputer on the training data
# imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy='constant', fill_value=0)
# imputer.fit(train_features_frame)
# train_features = imputer.transform(train_features_frame)

# # Data Imputer on the test data
# imputer.fit(test_features_frame)
# test_features = imputer.transform(test_features_frame)

# print(SimpleImputer(strategy='mean').fit(train_features_frame).statistics_)
# print(X[0])


# How many entries are missing in the original dataframe?
# for i in range(train_features.shape[1]):
#   n_miss = train_features_frame.iloc[:, i].isnull().sum()
#   percent = n_miss / train_features_frame.shape[0] * 100
#   print('> %d, Missing: %d (%.1f%%)'%(i, n_miss, percent))

# Transforming the array s.t. each patient only has one row (all features are concatenated)
# print(train_features.shape)
# train_features_transformed = train_features.reshape((18995, -1))
# print(train_features.shape)
# print(train_features[0])

def colToMatrix(array, colId):
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


# For the final output
baseExcessTest = colToMatrix(test_features, 10)
fibrinogeneTest = colToMatrix(test_features, 12)
astTest = colToMatrix(test_features, 17)
alkTest = colToMatrix(test_features, 27)
bilTotTest = colToMatrix(test_features, 33)
lactTest= colToMatrix(test_features, 6)
tropoTest = colToMatrix(test_features, 34)
sao2Test = colToMatrix(test_features, 20)
bilDirTest = colToMatrix(test_features, 29)
etco2Test = colToMatrix(test_features, 3)

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

# # Some sanity checks
# print('\n#########')
# print(f'(Test Data) Xst.shape: {Xst.shape}')
# print(f'(Training Data) Xs_train.shape: {Xs_train.shape}')
# print(f'(Training Data) Xs_test.shape: {Xs_test.shape}\n')
# print(f'The dimensions of the training data add up: {Xs_train.shape[1] + Xs_test.shape[1] == train_features.shape[0]/12}')
# print('#########\n')

# meanAcc = 0
# tot_prob = 0


class LogReg(nn.Module):
        def __init__(self, n_input_features):
            super(LogReg,self).__init__()
            self.layer1 = nn.Linear(n_input_features, 20)
            self.layer2 = nn.Linear(20, 10)
            self.layer3 = nn.Linear(10, 5)
            self.layer4 = nn.Linear(5, 10)
            self.layer5 = nn.Linear(10, 20)
            self.layer6 = nn.Linear(20, 1)

        def forward(self, x):
            layer1_output = torch.relu(self.layer1(x))
            layer2_output = torch.relu(self.layer2(layer1_output))
            layer3_output = torch.relu(self.layer3(layer2_output))
            layer4_output = torch.relu(self.layer4(layer3_output))
            layer5_output = torch.relu(self.layer5(layer4_output))
            layer6_output = torch.sigmoid(self.layer6(layer5_output))

            self.probabilities = layer6_output
            y_predicted = torch.sigmoid(self.probabilities)
            return y_predicted

#######################################################################################################################
### Sub Task 1

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
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_iters):
        # Predict
        # y_pred = forward(Xs_train[0])
        y_pred = model(Xs_train[j])

        # Compute loss
        loss = loss_function(y_pred.squeeze(), ys_train[0])

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

    # with torch.no_grad():
    #     y_predicted = model(Xs_test[j])
    #     # print(y_predicted.shape)
    #     # tot_prob += y_predicted[0].item()
    #     y_predicted = y_predicted.squeeze()
    #     # res_t1[j] = model.probabilities # TODO: gather them up reasonably. model.probabilities requires the call "model(MY_DESIRED_INPUT_DATA)" first
    #
    #     y_predicted_cls = y_predicted.round()
    #     acc = y_predicted_cls.eq(ys_test[j]).sum()/float(ys_test[j].shape[0])
    #     print(f'Accuracy: {acc*100:.3f} %')
    #     meanAcc = meanAcc + acc
    #
    #     # model(x_evaluation)
    #     # y_probabities = model.probabilities

    with torch.no_grad():
        y_predicted = model(Xst[j])
        # print(y_predicted.shape)
        # tot_prob += y_predicted[0].item()
        y_predicted = y_predicted.squeeze()
        # print(model.probabilities.shape)
        # print(res_t1.shape)
        res_t1[:, j] = model.probabilities.squeeze()     # TODO: gather them up reasonably. model.probabilities requires the call "model(MY_DESIRED_INPUT_DATA)" first

        # model(x_evaluation)
        # y_probabities = model.probabilities


# print(f'Average: {((meanAcc / act_col).item())*100:.3f} %')
# print(res_t1)

# export the results in the csv file
np.savetxt('res_2_sub1.csv', res_t1, delimiter=',', fmt='%.3f')

# #######################################################################################################################
# # Sub Task 2
#
print('begin task 2')

tempTab = colToMatrix(train_features, 7)
heartTab = colToMatrix(train_features, 32)
rrateTab = colToMatrix(train_features, 11)
wbcTab = colToMatrix(train_features, 14)

tempTest = colToMatrix(test_features, 7)
heartTest = colToMatrix(test_features, 32)
rrateTest = colToMatrix(test_features, 11)
wbcTest = colToMatrix(test_features, 14)

refSepsis = ground_truth[ :,11]

X_train_t2 = np.array([tempTab, heartTab, rrateTab, wbcTab])
X_test_t2 = np.array([tempTest, heartTest, rrateTest, wbcTest])
y_t2 = np.array([refSepsis])

# scaler = StandardScaler()
# X_train_t2 = scaler.fit_transform(X_train_t2)
# X_test_t2 = scaler.transform(X_test_t2)

X_train_t2 = torch.tensor(X_train_t2, dtype=torch.float32)
X_test_t2 = torch.tensor(X_test_t2, dtype=torch.float32)
y_t2 = torch.tensor(y_t2, dtype=torch.float32)


# Array to store the results for subtask 2
res_t2 = np.zeros(shape=(X_test_t2.shape[1], 0))
# print(res_t2.shape)

# Get size
print(f'X_train_t2.shape={X_train_t2.shape}')
n_samples = X_train_t2.shape[1]
n_features = X_train_t2.shape[0]

# Initialize weights
# w = torch.tensor(12*[0.0]  , dtype=torch.float32, requires_grad=True)

# Define Model
input_size = n_features
output_size = 1

model = LogReg(input_size)

#Training
n_iters = 100
learning_rate = 0.01

# loss = MSE
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # Predict
    # TODO: We need to figure out if we can load 4 features at the same time, maybe change the model for that?
    y_pred = model(X_train_t2)

    # Compute loss
    loss = loss_function(y_pred.squeeze(), y_t2[0])

    # Compute gradients
    loss.backward() #dloss/dx

    # Update weight
    # This makes sure gradient isn't part of the computational graph
    #with torch.no_grad():
    #    w -= learning_rate*w.grad
    optimizer.step()

    # We need to reinitialize weight not to accumulate
    #w.grad.zero_()
    optimizer.zero_grad()


with torch.no_grad():
    y_predicted = model(X_test_t2)
    # print(y_predicted.shape)
    # tot_prob += y_predicted[0].item()
    y_predicted = y_predicted.squeeze()
    print(model.probabilities.shape)
    print(res_t2.shape)
    res_t2[:] = model.probabilities.squeeze()

    # model(x_evaluation)
    # y_probabities = model.probabilities

# print(f'Average: {((meanAcc / act_col).item())*100:.3f} %')
print(res_t2)

# export the results in the csv file
np.savetxt('res_2_sub2.csv', res_t2, delimiter=',', fmt='%.3f')

