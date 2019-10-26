import os
import re
import sys
import random
import numpy as np

# tf.keras definition
# these may change depending on your install environments
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import multiply, add, Input, Dense, Dropout
from tensorflow.contrib.keras import backend as K
from tensorflow.python.keras.layers.merge import _Merge

import macro

# for W-GAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
# in this paper, \hat{x} = \eps x + \eps \tilde{x}
# \tilde{x} : generated sample, x : sample in dataset
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        eps = K.random_uniform((macro._BATCH_SIZE,1,1,1))
        return (eps*inputs[0]+(1-eps)*inputs[1])

# wasserstein loss for W-GAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

# gradient penalty for W-GAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
def gradient_penalty(y_true, y_pred, interpolate, lamb):
    grad = K.gradients(y_pred, interpolate)[0]
    norm = K.square(grad)
    norm_sum = K.sum(norm,axis=np.arange(1,len(norm.shape)))
    l2_norm = K.sqrt(norm_sum)
    gp_reg = lamb*K.square(1-l2_norm)
    return K.mean(gp_reg)

# generator model
def generator(data_shape):
    model = Sequential()
    model.add(Dense(macro._LAYER_DIM, activation='relu', input_shape=(macro._NOISE_DIM+macro._PROP_DIM,)))
    #model.add(Dropout(0.2))
    model.add(Dense(2*macro._LAYER_DIM, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(3*macro._LAYER_DIM, activation='relu'))
    #model.add(Dropout(0.2))
    # use sigmoid function to constrain output from 0 to 1.
    model.add(Dense(data_shape, activation='sigmoid'))
    return model

# discriminator model (from input to highest hidden layer)
def discriminator(data_shape):
    model = Sequential()
    model.add(Dense(2*macro._LAYER_DIM, activation='relu', input_shape=(data_shape,)))
    #model.add(Dropout(0.2))
    #model.add(Dense(3*macro._LAYER_DIM, activation='relu', input_shape=(data_shape,)))
    #model.add(Dropout(0.2))
    #model.add(Dense(2*macro._LAYER_DIM, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(macro._LAYER_DIM, activation='relu'))
    #model.add(Dropout(0.2))
    return model

# last layer of discriminator for classifying real or fake sample
def real_fake():
    model = Sequential()
    model.add(Dense(1, name='real_fake', input_shape=(macro._LAYER_DIM,)))
    return model

# last layer of discriminator for predicting properties
def prediction():
    model = Sequential()
    model.add(Dense(macro._PROP_DIM, name='predictions', input_shape=(macro._LAYER_DIM,)))
    return model
