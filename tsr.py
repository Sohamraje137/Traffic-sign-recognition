import pickle
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plot
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#import cv2
import hashlib
import os
from urllib.request import urlretrieve
from PIL import Image
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile
from PIL import Image
import matplotlib.gridspec as gridspec

from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import glob, os
import random

import numpy as np

#matplotlib inline
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras import models
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

# Load pickled data
import pickle



training_file ="train.p"
#validation_file="valiation.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
# with open(validation_file, mode='rb') as f:
#     valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
print(train.keys())
print(train['features'].shape)
X_train, y_train = train['features'], train['labels']
# X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']




# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
#n_validation = ?

# TODO: Number of testing examples.
n_test = X_test.shape[0]

image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#Y'=0.299R'+0.587G'+0.114B'

def grayscale(array):
    greyscale_data = []

    for i in range(0,array.shape[0]):
        image = array[i]    
        red,green,blue =image[:,:,0], image[:,:,1], image[:,:,2]
        grey_image = 0.2989 * red + 0.5870 *green  + 0.1140 *blue
        greyscale_data.append(grey_image)
        
        
    return np.array(greyscale_data)

X_train = grayscale(X_train)
X_test = grayscale(X_test)

print (X_train.shape)
print (X_test.shape)

def normalize_greyscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # ToDo: Implement Min-Max scaling for greyscale image data
    #costants
    a = 0.1
    b = 0.9
    xmin = 0
    xmax = 255
    
    x = image_data    
    x_prime = a + ((x-xmin)*(b-a))/(xmax-xmin)
    
    return x_prime



X_train = normalize_greyscale(X_train)
X_test = normalize_greyscale(X_test)
X_train=X_train.reshape(X_train.shape[0],32,32,1)
X_test=X_test.reshape(X_test.shape[0],32,32,1)


print (X_train.shape)
print (X_test.shape)

# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32

y_train = y_train.astype(np.float32)

y_test = y_test.astype(np.float32)
print (y_train.shape)
print (y_test.shape)
print (X_train.shape)
print (X_test.shape)

# Get randomized datasets for training and validation

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,test_size=0.15,random_state=832289)

print('Training features and labels randomized and split.')

# define model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32,32,1),kernel_regularizer=regularizers.l2(0)))
#model.add(Conv2D(32, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
#model.add(Conv2D(64, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
#model.add(Conv2D(128, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))

model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

'''
model.add(Conv2D(256, (3, 3),padding='same',kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))  '''
# model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

# Fully connected layer

model.add(Dense(43,kernel_regularizer=regularizers.l2(0)))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


model.fit(X_train,y_train,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_data=(X_valid,y_valid))


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model7')
