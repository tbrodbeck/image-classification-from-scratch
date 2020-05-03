from datetime import datetime
from keras.activations import relu
from keras.layers import Activation, Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import pickle
import sys
import tensorflow as tf

imagenettePath = 'imagenette2/'
dataset = 'imagenette'

""" parsing command line input """
# parsing command-line imputs
hyperparameterChoice = int(sys.argv[1])
runNr = sys.argv[2]

""" the hyperparameters """

epochs = 10
# different parameter configurations can be tracked in dictionaries for quick comparisons
if hyperparameterChoice==0: # test configuration with one epoch only
    hyperparameter = {
        'batchSize': 128,
        'keepProb': .5,
        'activationConv': 'relu',
        'activationDense': 'relu',
        'denseLayers': 1,
        'batchNorm': False,
    }
elif hyperparameterChoice==1:
    hyperparameter = {
        'batchSize': 128,
        'keepProb': .5,
        'activationConv': 'relu',
        'activationDense': 'relu',
        'denseLayers': 1,
        'batchNorm': False,
    }
elif hyperparameterChoice==2:
    hyperparameter = {
        'batchSize': 128,
        'keepProb': .25,
        'activationConv': 'relu',
        'activationDense': 'relu',
        'denseLayers': 2,
        'batchNorm': True,
    }
elif hyperparameterChoice==3:
    hyperparameter = {
        'imageSize': 224,
        'batchSize': 128,
        'keepProb': .5,
        'activationConv': 'relu',
        'activationDense': 'relu',
        'denseLayers': 1,
        'batchNorm': False,
    }

print('Chosen Hyperparameters:', hyperparameter)

batchSize = hyperparameter['batchSize']
keepProb = hyperparameter['keepProb']
activationConv = hyperparameter['activationConv']
activationDense = hyperparameter['activationDense']
# epochs = hyperparameter['epochs']
denseLayers = hyperparameter['denseLayers']
batchNorm = hyperparameter['batchNorm']

""" the loading and preprocessing the data """
# setup image preprocessing
datagen = ImageDataGenerator(horizontal_flip=True, 
                             rescale=1./255, # scales the images to values between 0 and 1
                             width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
                             rotation_range=1 # rotate the images randomly in the range of 1 degree
                            )

if dataset=='cifar10': # downloads CIFAR10 from keras
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    print(x_train.shape[0], 'Train Samples')
    print(x_test.shape[0], 'Test Samples')

    datagen.fit(x_train)
    train = datagen.flow(x_train, y_train, batch_size=batchSize, shuffle=True)
    val = (x_test, y_test)
    
elif dataset=='imagenette': # installs `imagenette` from disk
    # load train data
    train = datagen.flow_from_directory("{}train/".format(imagenettePath), class_mode="categorical", shuffle=True, batch_size=batchSize, target_size=(224, 224))
    # load val data
    val = datagen.flow_from_directory("{}val/".format(imagenettePath), class_mode="categorical", shuffle=True, batch_size=batchSize, target_size=(224, 224))
else:
    raise ValueError(f'Invalid Value: {dataset} - only cifar10 and imagenette are valid')
    
    
""" the architectures """

if dataset=='imagenette':
    architecture = 1 # Architecture 1 is tuned to perform well on the imagenette dataset (but it does not perform too well on CIFAR)
if dataset=='cifar10':
    architecture = 2 # Architecture 2 performs better for CIFAR, but has a training time that is too large for 'imagenette' (more than 1h per epoch on my resources)

if architecture==1:
    model = Sequential()

    # 1st convolution block
    model.add(Conv2D(16, (5, 5), input_shape=train[0][0].shape[1:], strides=(1, 1), padding='same'))
    model.add(Activation(activationConv))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    # 2nd convolution block
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(Activation(activationConv))
    model.add(MaxPool2D(pool_size=(2, 2)))
    if batchNorm:
        model.add(BatchNormalization())

    # 3rd convolution block
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation(activationConv))
    model.add(MaxPool2D(pool_size=(2, 2)))
    if batchNorm:
        model.add(BatchNormalization())
    model.add(Dropout(keepProb))
    
    # Dense block
    model.add(Flatten())
    for i in range(denseLayers):
        model.add(Dense(units=100, activation=activationDense))
    model.add(Dropout(keepProb))

    # Output layer
    model.add(Dense(units=10))
    model.add(Activation('softmax'))

elif architecture==2:
    model = Sequential()
    # 1st 2xconv block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=train[0][0].shape[1:]))
    model.add(Activation(activationConv))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation(activationConv))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(keepProb))

    # 2nd 2xconv block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation(activationConv))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activationConv))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(keepProb))

    # ANN block
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(units=10))
    model.add(Activation('softmax'))

else:
    raise ValueError(f'architecture number {architecture} is out of range')
    
model.summary()
# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

""" the training """
# fit on data
result = model.fit_generator(train, epochs=epochs, validation_data=val)

outDir = 'searchResults/'
if not os.path.exists(outDir):
    os.makedirs(outDir)
runName = '{}{}_{}'.format(outDir, runNr, hyperparameterChoice)
model.save('{}.model'.format(runName))
with open('{}result.p'.format(runName), 'wb') as file:
    pickle.dump(result, file)
print(f'Models and Training Output were saved to the Directory {outDir}')
