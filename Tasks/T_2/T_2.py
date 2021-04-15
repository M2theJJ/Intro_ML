# Some general information:
# We got 18995 patients.
# Each patient has 12 measurements, thus in total we got 227940 measurements (test_features.csv)



### This code gives a metric on how many data points are missing in each feature
from pandas import read_csv
dataframe = read_csv('train_features_naively_imputed.csv', header=0)

for i in range(dataframe.shape[1]):
	n_miss = dataframe.iloc[:, i].isnull().sum()
	percent = n_miss / dataframe.shape[0] * 100
	print('> %d, Missing: %d (%.1f%%)'%(i, n_miss, percent))



### Idea: I make two assumptions. The first one is, that the patients are independent, thus the measurements of one
### patient does not influence any other patient. The other is, if no measurements of a certain feature was done, it
### is probably neglictable and thus can be estimated to be around the average.
### Thus the code looks at the twelve measurements for any patient. If there is any data point for any of the features,
### then the missing data points will be an average of the one that was measured. If all the points are missing, then
### the measurements will be imputed by the healthy human average of that feature.

### Result: The impute function is stupendeously slow and with that, it is not of any real use to us!
### Nevertheless I have run it, and the results can still be used for the tasks at hand.


import numpy as np
import random as rnd
from pandas import read_csv

dataframe = read_csv('train_features.csv', header=0)

# For every patient, go though the 12 rows of data and average the entries
# 	If there is an average (i.e. at least one point was measured), then take that to fill in the nan's of that column
#	If there is no average, then fill the column with values picked randomly from the corresponding 'average_values' array

def impute(patient):
	# pid,	Time,	Age,	EtCO2,	PTT,	BUN,	Lactate,	Temp,	Hgb,		HCO3,	BaseExcess,	RRate*,	Fibrinogen,	Phosphate,	WBC,	Creatinine,	PaCO2,	AST,	FiO2,	Platelets,	SaO2,	Glucose,	ABPm*,	Magnesium,	Potassium,	ABPd,	Calcium,	Alkalinephos,	SpO2,	Bilirubin_direct,	Chloride,	Hct,	Heartrate,	Bilirubin_total,	TroponinI,	ABPs,	pH
	# -		-		-		35-45	25-35	7-20	0.8-1.5		36-38	12-17.5		22-28	(-2)-2		0?		150-400		2.5-4.5		4.5-11	0.5-1.2		38-42	5-40	0.21	150-450		97-100	72-140		??		1.8-2.2		3.6-5.2		??		8.6-10.3	20-140			95-100	0-0.3				96-106		36-50	60-100		0.1-1.2				0-0.4		??		7.35-7.45
	# * For RRate, ABPm, ABPdm, and  ABPs I could not find good data, but it looks like those features are pretty good populated, anyway
	# The data was found via quick Google search and might not be accurate.
	average_values = [[35,45], [25,35], [7,20], [0.8,1.5], [36,38], [12,17.5], [22,28], [-2,2], [0,0], [150,400], [2.5,4.5], [4.5,11], [0.5,1.2], [38,42], [5,40], [0.21,0.21], [150,450], [97,100], [72,140], [0,0], [1.8,2.2], [3.6,5.2], [0,0], [8.6,10.3], [20,140], [95,100], [0,0.3], [96,106], [36,50],  [60,100], [0.1,1.2], [0,0.4], [0,0], [7.35,7.45]]

	# Create a list of lists
	average_patient_values = np.empty((len(average_values), 0)).tolist()
	# Go though all the 12 data rows of a patient and save all the available information into the corresponding list (nan's are skipped)
	for i in range(12):
		for j in range(dataframe.shape[1]-3):
			if np.isnan(dataframe.iloc[(i+patient), j+3]):
				continue
			else:
				average_patient_values[j].append(dataframe.iloc[(i+patient), j+3])

	# Calculate the average of the lists. Empty lists (i.e. originally only 'nan') are set to inf.
	for i in range(len(average_patient_values)):
		if not average_patient_values[i]:
			average_patient_values[i] = np.inf
		else:
			average_patient_values[i] = np.mean(average_patient_values[i])

	# Impute data: Again, go through all 12 data rows of a patient. If a 'nan' occurs, then see if there is an average available,
	# If no average exists (i.e. all the values of this column are 'nan', then generate a random value from the average interval.
	# Else, choose the average value that was computed before.
	for i in range(12):
		for j in range(dataframe.shape[1]-3):
			if np.isnan(dataframe.iloc[(i+patient), j+3]) and average_patient_values[j] == np.inf:
				dataframe.iloc[(i+patient), j+3] = rnd.uniform(average_values[0][0], average_values[0][1])
			elif np.isnan(dataframe.iloc[(i+patient), j+3]):
				dataframe.iloc[(i+patient), j+3] = average_patient_values[j]


# Call the imputation function for each of the patients. The stride is 12, as each patient has 12 data rows.
# for i in range(18995):
# 	print('> Imputing data for patient %d (line %d to %d)'%(dataframe.iloc[i*12,0], i*12+1, i*12+12))
# 	impute(i*12)
#
#
# # This code writes a pandas dataframe directly into a zip file, supposed df is a pandas dataframe containing the result
# dataframe.to_csv('train_features_naively_imputed.csv', float_format='%.3f', header=True, index=False)



# We could try to impute the missing data with the iterative imputer class from scikit-learn:
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
# I got inspired by this website when it came to the imputation for our dataset
# https://machinelearningmastery.com/iterative-imputation-for-missing-values-in-machine-learning/

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
# np.savetxt('train_features_automatically_imputed.csv', Xtrans, delimiter=',', fmt='%1.2f')


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
