import numpy as np
import  os
import csv
from collections import Counter
import random
import math
import copy
import math
from sklearn import svm,metrics
from multiprocessing import Pool
import time


#Defines the directory containing the test_set.csv, training_set.csv, validation_set.csv and their labels
#Specify the path where files are located
path=".\\ANN"
#Sets the path as its working directory
os.chdir(path)



test_set='test_set.csv'
train_set='training_set.csv'
train_label='training_labels.csv'
validation_set='validation_set.csv'
validation_label='validation_labels.csv'



#Imports all the related csv files for training, validating and predicting for the SVM classifiers

with open(test_set,'r') as test_file:
	read_data=csv.reader(test_file,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
	test_data_list=[np.array(data) for data in read_data]
	test_data_array=np.reshape(test_data_list,(len(test_data_list),354))
	
with open(test_set,'r') as test_file:
	read_data=csv.reader(test_file,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
	test_data_list=[np.array(data) for data in read_data]

with open(train_set,'r') as train_file:
	read_data=csv.reader(train_file,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
	train_data=[np.array(data) for data in read_data]

with open(train_label,'r') as trainLab_file:
	read_data=csv.reader(trainLab_file,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
	train_labels=[int(np.array(data)) for data in read_data]


with open(validation_set,'r') as test_file:
	read_data=csv.reader(test_file,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
	validation_data=[np.array(data) for data in read_data]

with open(validation_label,'r') as valid_label:
	read_data=csv.reader(valid_label,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
	validation_labels=[int(np.array(data)) for data in read_data]


#The data are reshaped and converted into array types for the SVM classifier to take as input
train_data=np.reshape(np.asarray(train_data),(len(train_data),354))
train_labels=np.array(train_labels)


validation_data=np.reshape(np.asarray(validation_data),(len(validation_data),354))
validation_labels=np.asarray(validation_labels)






#The SVM classifiers imported via the scikit-learn module
#The kernels used are 'linear' and 'rbf'
clf_rbf=svm.SVC(kernel='rbf',C=100	,decision_function_shape='ovo',gamma=2.0)
clf_linear=svm.LinearSVC(C=55)


#The training of the SVM models
clf_rbf.fit(train_data,train_labels)
clf_linear.fit(train_data,train_labels)


#Prediction of the SVM models for both the training and validation data points
validation_predict_rbf=clf_rbf.predict(validation_data)
training_predict1_rbf=clf_rbf.predict(train_data)


validation_predict_linear=clf_linear.predict(validation_data)
training_predict_linear=clf_linear.predict(train_data)


#After training and validation the LinearSVM is used to predict the class of the datapoints in the test_set.csv
test_predict=clf_linear.predict(test_data_array)



#Prints the accuracy of the training and validation phase
print("Accuracy: ",metrics.accuracy_score(validation_labels,validation_predict_linear))
print("Accuracy: Training",metrics.accuracy_score(train_labels,training_predict_linear))


print("Accuracy: ",metrics.accuracy_score(validation_labels,validation_predict_rbf))
print("Accuracy: Training",metrics.accuracy_score(train_labels,training_predict1_rbf))


#Function is used to write the predicted class of the test data points to a csv file
def write_prediction_svm(test_predict):
	with open('predicted_svm.csv','w',newline="") as file:
	    predicted_writer=csv.writer(file,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
	    for Class in test_predict:
	        predicted_writer.writerow([Class])		

write_prediction_svm(test_predict)


