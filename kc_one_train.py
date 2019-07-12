import keras;
from keras.models import Sequential;
from keras.layers import Dense;
from time import time
import numpy as np
import csv
import matplotlib.pyplot as plt

#Complete dataser
dataset = np.loadtxt('dataset.csv',delimiter=',')

#Dataset used for validation
val_dataset = np.loadtxt('val_data.csv', delimiter=',')

#dataset used for testing the model
test_dataset = np.loadtxt('test_data.csv', delimiter=',')

#Training data slices
X_Train = dataset[:,0:32]
Y_Train = dataset[:,32]

#Validation data slices
X_val = val_dataset[:,0:32]
Y_val = val_dataset[:,32]

#data which can be used for testing
X_Test = test_dataset[:,0:32]

#Sequential model allows all layers to be added sequentially.
network = Sequential();

#Input neurons: 32 and first layer neurons is 6
layer1 = Dense(units=6,
                  activation='relu',
                  kernel_initializer='uniform',
                  input_dim=32)

network.add(layer1);

layer2 = Dense(units=1,
                  activation='relu',
                  kernel_initializer='uniform')


network.add(layer2);

#Compiling the model with required parameters
network.compile(optimizer='adam',
		loss='mse',
		metrics=['accuracy'])

#History which can be used to draw visualization graphs 
history = network.fit(X_Train, Y_Train, validation_data=(X_val, Y_val), 
          epochs=50, batch_size=20)

#Saving model for usage
network.save("model1.h5")

from ann_visualizer.visualize import ann_viz;

#This is a visualization tool
ann_viz(network, title="");

weights1 = layer1.get_weights()[0]
weights2 = layer2.get_weights()[0]

print(weights1)
print(weights2)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()