import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from numpy import asarray
from numpy import zeros
from numpy import ones
from math import sqrt
from numpy import load
from numpy.random import randn, randint

import os
import shutil

from PIL import Image
from skimage.transform import resize
import cv2

import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, AveragePooling2D, LeakyReLU, BatchNormalization, Add, Input, UpSampling2D, Reshape, Layer
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend




# weighted sum output
class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha = 0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')
    # output a weighted sum of inputs
    def merge_function(self, inputs):
        assert(len(inputs) == 2)
        output = ((1.0-self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

# mini-batch standard deviation layer
class MinibatchStdev(Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev,self).__init__(**kwargs)

    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = backend.mean(inputs, axis = 0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = backend.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims = True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        stdev = backend.sqrt(mean_sq_diff)
        mean_pix = backend.mean(stdev, keepdims=True)
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

        combined = backend.concatenate([inputs, output], axis =- 1)

        return combined
    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        return tuple(input_shape)


# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
    def call(self, inputs):
        values = inputs**2.0
        mean_values = backend.mean(values, axis = 1, keepdims = True)
        mean_values += 1.0e-8
        l2 = backend.sqrt(mean_values)
        normalized = inputs / l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape

# calculate wasserstein loss
def wgan_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

# add a discriminator block
def add_descriminator_block(old_model, n_input_layers=3):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    in_shape = list(old_model.input.shape)
    input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)

    d = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(0.2)(d)
    d = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(0.2)(d)
    d = Conv2D(128, (1, 1), padding = 'same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(.2)(d)
    d = AveragePooling2D()(d)
    block_new = d

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model1 = Model(in_image, d)
    model1.compile(loss=wgan_loss, optimizer=Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    downsample=AveragePooling2D()(in_image)

    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)

    d = WeightedSum()([block_old, block_new])

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model2 = Model(in_image, d)
    model2.compile(loss=wgan_loss, optimizer=Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    return [model1, model2]
# define the discriminator models for each image resolution
def define_descriminator(n_blocks, input_shape=(4,4,3)):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    model_list = list()
    in_image = Input(shape=input_shape)
    d = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = MinibatchStdev()(d)
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=.2)(d)
    d = Flatten()(d)
    out_class = Dense(1)(d)
    model = Model(in_image, out_class)

    model.compile(loss=wgan_loss, optimizer=Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    model_list.append([model, model])
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_descriminator_block(old_model)
        model_list.append(models)
    return model_list

# add a generator block
def add_generator_block(old_model):
    init = RandomNormal(stddev=0.02)
    const = max_norm(max_value=1.0)
    block_end = old_model.layers[-2].output
    upsampling = UpSampling2D()(block_end)
    g = Conv2D(128, (3, 3), padding = 'same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3, 3), padding = 'same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    out_image = Conv2D(3, (1, 1), padding = 'same', kernel_initializer=init, kernel_constraint=const)(g)
    model1 = Model(old_model.input, out_image)
    out_old = old_model.layers[-1]
    out_image2 = out_old(upsampling)
    merged = WeightedSum()([out_image2, out_image])
    model2 = Model(old_model.input, merged)
    return [model1, model2]

# define generator models
def define_generator(latent_dim, n_blocks, in_dim = 4):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    model_list = []
    in_latent = Input(shape=(latent_dim, ))
    g = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = Reshape((in_dim, in_dim, 128))(g)
    g = Conv2D(128, (3, 3), padding = 'same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(0.2)(g)
    g = Conv2D(128, (3, 3), padding = 'same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(0.2)(g)
    out_image = Conv2D(3, (1, 1), padding = 'same', kernel_initializer=init, kernel_constraint=const)(g)
    model = Model(in_latent, out_image)
    model_list.append([model, model])
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_generator_block(old_model)
        model_list.append(models)
    return model_list

# define composite models for training generators via discriminators
def define_composite(discriminators, generators):
    model_list = []
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]
        d_models[0].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wgan_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wgan_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        model_list.append([model1, model2])
    return model_list
