# coding:utf-8
import os
from keras.models import save_model
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout, concatenate, multiply
from keras.layers import LSTM
from keras.utils.io_utils import HDF5Matrix
import h5py
#from exp_models.my_callback import *
import time

# global constants
DROPOUT = 0.2

def create_model(image_dim, audio_vector_dim, learning_rate, weight_init,output_dim,optimizer):

    # Define image input layer
    (img_rows, img_cols, img_channels) = image_dim
    input_spacetime_img = Input(shape=(img_rows, img_cols, img_channels))
    input_rgb_img = Input(shape=(img_rows, img_cols, img_channels))

    # Channel 1 - Conv Net Layer
    x = Conv2D(48, 11, strides=4, activation='relu', padding='same')(input_spacetime_img)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Channel 2 - Conv Net Layer 1
    y = Conv2D(48, 11, strides=4, activation='relu', padding='same')(input_rgb_img)
    y = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)
    y = BatchNormalization()(y)

    '''
    x = conv2D_bn(img_input, 3, 11, 11, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(strides=(4, 4), pool_size=(4, 4), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    y = conv2D_bn(img_input, 3, 11, 11, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(strides=(4, 4), pool_size=(4, 4), dim_ordering=DIM_ORDERING)(y)
    y = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y)
    '''
    '''
    # Channel 1 - Conv Net Layer 2
    x = conv2D_bn(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 2 - Conv Net Layer 2
    y = conv2D_bn(y, 48, 55, 55, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(y)
    y = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y)
    '''

    # Channel 1 - Conv Net Layer 3
    x = Conv2D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Channel 2 - Conv Net Layer 3
    y = Conv2D(128, 5, activation='relu', padding='same')(y)
    y = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)
    y = BatchNormalization()(y)
    '''
    x = conv2D_bn(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    y = conv2D_bn(y, 128, 27, 27, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(y)
    y = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y)
    '''
    # Channel 1 - Conv Net Layer 4
    x = Conv2D(192, 3, activation='relu', padding='same')(x)

    # Channel 2 - Conv Net Layer 4
    y = Conv2D(192, 3, activation='relu', padding='same')(y)
    '''
    x1 = merge([x, y], mode='concat', concat_axis=CONCAT_AXIS)
    x1 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x1)
    x1 = conv2D_bn(x1, 192, 13, 13, subsample=(1, 1), border_mode='same')
    y1 = merge([x, y], mode='concat', concat_axis=CONCAT_AXIS)
    y1 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y1)
    y1 = conv2D_bn(y1, 192, 13, 13, subsample=(1, 1), border_mode='same')
    '''
    # Channel 1 - Conv Net Layer 5
    x = Conv2D(192, 3, activation='relu', padding='same')(x)

    # Channel 2 - Conv Net Layer 5
    y = Conv2D(192, 3, activation='relu', padding='same')(y)
    '''
    y2 = merge([x1, y1], mode='concat', concat_axis=CONCAT_AXIS)
    y2 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y2)
    y2 = conv2D_bn(y2, 192, 13, 13, subsample=(1, 1), border_mode='same')

    x2 = merge([x1, y1], mode='concat', concat_axis=CONCAT_AXIS)
    x2 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x2)
    x2 = conv2D_bn(x2, 192, 13, 13, subsample=(1, 1), border_mode='same')
    '''

    # Channel 1 - Cov Net Layer 6
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    # Channel 2 - Cov Net Layer 6
    y = Conv2D(128, 3, activation='relu', padding='same')(y)
    y = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)
    y = BatchNormalization()(y)
    '''
    x3 = conv2D_bn(x2, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x3 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(x3)
    x3 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x3)

    y3 = conv2D_bn(y2, 128, 27, 27, subsample=(1, 1), border_mode='same')
    y3 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(y3)
    y3 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y3)
    '''

    # Channel 1 - Cov Net Layer 7
    x4 = multiply([x, y])
    x4 = Flatten()(x4)
    x4 = Dense(2048, activation='relu')(x4)
    x4 = Dropout(DROPOUT)(x4)

    # Channel 2 - Cov Net Layer 7
    y4 = multiply([x, y])
    y4 = Flatten()(y4)
    y4 = Dense(2048, activation='relu')(y4)
    y4 = Dropout(DROPOUT)(y4)

    '''
    x4 = merge([x3, y3], mode='mul', concat_axis=CONCAT_AXIS)
    x4 = Flatten()(x4)
    x4 = Dense(2048, activation='relu')(x4)
    x4 = Dropout(DROPOUT)(x4)

    y4 = merge([x3, y3], mode='mul', concat_axis=CONCAT_AXIS)
    y4 = Flatten()(y4)
    y4 = Dense(2048, activation='relu')(y4)
    y4 = Dropout(DROPOUT)(y4)
    '''
    # Channel 1 - Cov Net Layer 8
    x5 = multiply([x4, y4])
    x5 = Dense(2048, activation='relu')(x5)
    x5 = Dropout(DROPOUT)(x5)

    # Channel 2 - Cov Net Layer 8
    y5 = multiply([x4, y4])
    y5 = Dense(2048, activation='relu')(y5)
    y5 = Dropout(DROPOUT)(y5)
    '''
    x5 = merge([x4, y4], mode='mul')
    x5 = Dense(2048, activation='relu')(x5)
    x5 = Dropout(DROPOUT)(x5)

    y5 = merge([x4, y4], mode='mul')
    y5 = Dense(2048, activation='relu')(y5)
    y5 = Dropout(DROPOUT)(y5)
    '''

    # Final Channel - Cov Net 9
    xy = concatenate([x5, y5])

    #y Note that LSTM expects input shape: (nb_samples, timesteps, feature_dim)
    xy = Reshape((1, 4096))(xy)
    xy = LSTM(256, input_shape=(1, 4096), dropout=0.2, return_sequences=True)(xy)
    xy = LSTM(256, dropout=0.2, name='LSTM_reg_output')(xy)
    network_output = Dense(output_dim)(xy)#最後にオーディオデータの次元数にあわせる

    model = Model(input=[input_spacetime_img,input_rgb_img], output=network_output)
    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd  = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    #print("learning rate:",learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(model.summary())
    return model
