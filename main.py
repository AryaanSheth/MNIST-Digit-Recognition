'''
Filename: MNISTclassification\main.py
Created Date: Wednesday, May 25th 2022, 3:00:07 pm
Author: Aryaan Sheth

Copyright (c) 2022 https://aryaan.dev
'''

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from keras.datasets import mnist    # DATA
"""
terminal 
pip install pipreqs
...
pipreqs -> requirements.txt
"""
np.random.seed(0)   # constant seed for reproducibility

"""
2022-05-25 15:12:06.950769: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-05-25 15:12:06.951222: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Install CUDA and cudnn from https://developer.nvidia.com/cuda-downloads
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data() # load data

"""
visualize data 
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
"""

classes = 10 # number of classes
# f, ax = plt.subplots(1, classes, figsize = (28, 28)) # subplots for each number in 0-9

"""
Tes Sample Visualizer
for i in range(0, classes): # plot each number in 0-9
    sample = x_train[y_train == i][0]   # get first sample of each number
    ax[i].imshow(sample, cmap = 'gray') # plot
    ax[i].set_title(i, fontsize = 10) # set title\
plt.show()
"""


y_train = keras.utils.to_categorical(y_train, classes) # one hot encode
y_test = keras.utils.to_categorical(y_test, classes)

"""
for i in range(10):
    print(y_train[i])   # prints vector of one hot encoded values for each number in 0-9 
"""

# Normalize Data
x_train = x_train/255.0
x_test = x_test/255.0

# Reshape Data
x_train = x_train.reshape(x_train.shape[0], -1) # (60000, (28, 28)) -> (60000, 784)
x_test = x_test.reshape(x_test.shape[0], -1)

# Create Model
model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu'))   # input layer
model.add(Dense(units=128, activation='relu'))# hidden layer
model.add(Dropout(0.25)) # deactivate 25% of neurons to prevent overfitting data
model.add(Dense(units=classes, activation='softmax')) # softmax activation function

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # compile model
model.summary()

# Train Model
batch_size = 512
epochs = 15
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test)) # train model


# Evaluate Model
test_loss, test_accuracy = model.evaluate(x_test, y_test,)
# print(f'Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f}') Test Loss: 0.06854455918073654 Test Accuracy: 0.9786

y_prediction = model.predict(x_test)
y_prediction_classes = np.argmax(y_prediction, axis=1)
#print(y_prediction, y_prediction_classes)

n = 1 # int(input('Enter Number of Predictions: '))

#######################
def example_prediction():   # plot example prediction
    global y_true
    # Generate Example Prediction
    rand_index = np.random.choice(random.randint(0, 9))
    x_sample = x_test[rand_index]
    y_true = np.argmax(y_test, axis=1)
    y_sample_true = y_true[rand_index]
    y_sample_prediction_classes = y_prediction_classes[rand_index]
    # Plot Sample Prediction
    plt.title(f'Prediction: {y_sample_prediction_classes} True: {y_sample_true}')   # plot
    plt.imshow(x_sample.reshape(28, 28), cmap='gray')

def confusion_matrix_plot(): # plot confusion matrix
    # Confusion Matrix
    confusion_mtx = confusion_matrix(y_true, y_prediction_classes)
    # Generates a heatmap of the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(confusion_mtx, annot=True, fmt = 'd', ax=ax, cmap='Greens') # plot
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
for i in range(n):  # plot n examples
    example_prediction() 
    confusion_matrix_plot()
    plt.show() # show plots
    