import keras;
from keras.models import Sequential;
from keras.layers import Dense;
from time import time
import numpy as np
import csv

#Dataset which we need to use for the model
test_dataset = np.loadtxt('test_data.csv', delimiter=',')

#Since the dataset has about 32 columns
X_Test = test_dataset[:,0:32]

#Place to save the trained model into. This format stores the weights, biases of the trained DL Model
network = keras.models.load_model('model1.h5')

#Prediction to get the required accuracy
prediction_value = network.predict(X_Test)

#To test the model with some inputs and store them into the dataset so that it helps to train the model later with updated data
for x in range(len(prediction_value)):
	print(prediction_value[x])
	if (prediction_value[x] == 0):
		value = raw_input("Would you like to insert?")
		if(value == "yes"):
			X_Writer_Test = np.hstack(X_Test, np.ones((X_Test.shape[0], 1), dtype=X_Test.dtype))
			f = open('dataset.csv','ab')
			np.savetxt(f,X_Writer_Test,delimiter=',')
			f.close()
		




