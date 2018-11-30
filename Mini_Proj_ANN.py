
import  os
import csv
from collections import Counter
import random
import math
import copy
import math
import time
import numpy as np






def validation_phase(w_h1,b_h1,w_h2,b_h2,w_out,b_out,Bool):

	train_data_m=np.load('traindocs.npy')
	train_labels_m=np.load('trainlabels.npy')
	test_data_m=np.load('testdocs.npy')
	test_labels_m=np.load('testlabels.npy')
	train_data=train_data_m[0]
	train_labels=train_labels_m[0]
	test_data=test_data_m[0]
	test_labels=test_labels_m[0]


	train_labels=train_labels.tolist()
	#train_labels=get_labels(train_labels)
	test_labels=test_labels.tolist()
	#test_labels=get_labels(test_labels)


	#When Bool is true the accuracy and error of the validation dataset is evaluated
	if Bool:
		validation_data=test_data
		validation_labels=test_labels
		#Use the global variables that were imported from the CSV files
		
		#Note that we instead use the index of each the dataset since it is easier to manipulate and control
		#We access the relevant data point or label via their index
	#When Bool is false the accuracy and error of the training dataset is evaluated
	#This is done so we can see if the model is already overfitting
	
	else:
		validation_data=train_data
		validation_labels=train_labels
	
	#Defines the architecture of the ANN, the same architecture as with the training phase
	n_input=100
	n_out=2
	correct=0
	incorrect=0
	totalerr=0

	
	for x_in,label in zip(validation_data,validation_labels):
		#Defines the input data point as well as the actual label of the data point in order to measure the accuracy and error
		x_in=np.reshape(x_in,(n_input,1))
		d_out=np.reshape(label,(n_out,1))
		#We create an (8,1) vector for the output vector
		#This vector has zero elements for all except at the index+1 that corresponds to the actual class of the data point
		#Hence if the data point has class 5, the value at index '4' of the vector (8,1) is 1 and the rest are zero

		#Note that the weights and biases used in this network are the output of the neural netowrk during training

		#Hidden Layer 1
		v_h1=np.dot(w_h1,x_in)+b_h1
		y_h1=np.divide(1,(1+np.exp(-v_h1)))

		#Hidden Layer2
		v_h2=np.dot(w_h2,y_h1)+b_h2
		y_h2=np.divide(1,(1+np.exp(-v_h2)))

		#Output Layer
		v_out=np.dot(w_out,y_h2)+b_out
		out=np.divide(1,(1+np.exp(-v_out)))
		err=d_out-out
		
		#Aggregates the error for each data point
		#The totalerr at the end of the iteration calculates the sum of squared error
		#This will be used for the calcuation of the MSE
		totalerr+=(np.sum(np.multiply(err,err)))
		
		#Outputs the index of the array with the highest in the output Array
		#But index+1 represents the class of the data point itself.
		index_class=np.argmax(out)
		
		if label.index(1.0)==index_class:
			correct+=1
			
		else:
			incorrect+=1

	#Calculates the MSE
	totalerr=totalerr/(len(validation_labels))
	accuracy=correct/(correct+incorrect)



	return accuracy,totalerr



def training_phase():
	
	train_data_m=np.load('traindocs.npy')
	train_labels_m=np.load('trainlabels.npy')
	
	
	train_data=train_data_m[0]
	train_labels=train_labels_m[0]
	

	#Define Architecture
	n_input=100
	n_h1=12
	n_h2=12	
	n_out=2
	#Learning Rate
	eta=0.2 #learning rate
	
	#Defines the maximum Epoch
	max_epoch=100	
	#Initialize weights	

	#Initalizes the weight by random sampling
	x_in=np.zeros((n_input,1))
	w_h1=-0.1+(0.1+0.1)*np.random.rand(n_h1,n_input)
	b_h1=-0.1+(0.1+0.1)*np.random.rand(n_h1,1)
	w_h2=-0.1+(0.1+0.1)*np.random.rand(n_h2,n_h1)
	b_h2=-0.1+(0.1+0.1)*np.random.rand(n_h2,1)
	w_out=-0.1+(0.1+0.1)*np.random.rand(n_out,n_h2)
	b_out=-0.1+(0.1+0.1)*np.random.rand(n_out,1)



	#Training phase
	for i in range(max_epoch):
		totalerr=0
		for x_in,label in zip(train_data,train_labels):

			d_out=np.reshape(label,(n_out,1))
			x_in=np.reshape(x_in,(100,1))


			
		#Forward pass
			#hidden layer-1
			v_h1=np.dot(w_h1,x_in)+b_h1
			
			y_h1=np.divide(1,(1+np.exp(-v_h1)))
			
			#hidden layer-2
			v_h2=np.dot(w_h2,y_h1)+b_h2
			
			y_h2=np.divide(1,(1+np.exp(-v_h2)))
			
			#output layer
			v_out=np.dot(w_out,y_h2)+b_out
			#print(v_out.shape)
			out=np.divide(1,(1+np.exp(-v_out)))
			
			#out1=np.zeros((2,1))
			#out1[(np.argmax(out))]=1
			

	#Back prpopagation Propagation
			err=d_out-out
			totalerr+=np.sum(err*err)
			
			delta_out=err*out*(1-out)
			delta_h2=y_h2*(1-y_h2)*(np.dot(w_out.T,delta_out))
			delta_h1=y_h1*(1-y_h1)*(np.dot(w_h2.T,delta_h2))

			w_out=w_out+eta*np.dot(delta_out,y_h2.T)
			b_out=b_out+eta*delta_out

			w_h2=w_h2+eta*np.dot(delta_h2,y_h1.T)
			b_h2=b_h2+eta*delta_h2
			
			w_h1=w_h1+eta*np.dot(delta_h1,x_in.T)
			b_h1=b_h1+eta*delta_h1
	
		#For every 10th epoch the accuracy and error of the model tested on the validation and training set are printed	
		if i%10==0:
			print(i)
			accuracy_train,error_train=validation_phase(w_h1,b_h1,w_h2,b_h2,w_out,b_out,False)
			accuracy_validation,error_validation=validation_phase(w_h1,b_h1,w_h2,b_h2,w_out,b_out,True)
			print('Train accuracy:',accuracy_train,'	Error Train:',error_train,)
			print('Validation accuracy:',accuracy_validation,'	Error Validation:',error_validation,)
		#The learning rate decreases every 200 by formula given below
		if i%200==0:
			eta=eta/(1+i/200)
		#Leanring rate eta increases by 10% but goes to back to its orginal value after the 100th epoch
		#eta=eta*.10
		#if i%100==0:
			#eta=base_eta

	#The weights and biases after finishing the training phase is printed to a text file	


	return w_h1,b_h1,w_h2,b_h2,w_out,b_out


if __name__=='__main__':
	training_phase()

