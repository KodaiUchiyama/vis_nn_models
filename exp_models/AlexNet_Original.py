"""
    Model Name:

        AlexNet - using the Functional Keras API

        Replicated from the Original AlexNet Paper

    Paper:

         ImageNet classification with deep convolutional neural networks by Krizhevsky et al. in NIPS 2012

    Alternative Example:

        Available at: http://caffe.berkeleyvision.org/model_zoo.html

        https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet

    Original Dataset:

        ILSVRC 2012

"""
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers import Input, BatchNormalization
from keras.models import Model


# global constants
NB_CLASS = 1000         # number of classes
LEARNING_RATE = 0.01
MOMENTUM = 0.9
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005   # L2 regularization factor
USE_BN = True           # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
#DIM_ORDERING = 'th'

'''
def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''

        Info:
            Function taken from the Inceptionv3.py script keras github


            Utility function to apply to a tensor a module conv + BN
            with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Conv2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x
'''

def create_model():

    # Define image input layer
    DIM_ORDERING == 'tf'
    INP_SHAPE = (224, 224, 3)  # 3 - Number of RGB Colours
    img_input = Input(shape=INP_SHAPE)
    CONCAT_AXIS = 3


    # Channel 1 - Conv Net Layer
    x = Conv2D(48, 11, strides=4, activation='relu', padding='same')(img_input)
    x = MaxPooling2D(3, strides=2)(x)
    x = BatchNormalization()(x)

    # Channel 2 - Conv Net Layer 1
    y = Conv2D(48, 11, strides=4, activation='relu', padding='same')(img_input)
    y = MaxPooling2D(3, strides=2)(y)
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
    x = Conv2D(128, 5, strides=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(3, strides=2)(x)
    x = BatchNormalization()(x)

    # Channel 2 - Conv Net Layer 3
    y = Conv2D(128, 5, strides=3, activation='relu', padding='same')(y)
    y = MaxPooling2D(3, strides=2)(y)
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
    x1 = keras.layers.concatenate([x, y])
    x1 = Conv2D(192, 3, strides=, activation='relu', padding='same')(x1)

    # Channel 2 - Conv Net Layer 4
    y1 = keras.layers.concatenate([x, y])
    y1 = Conv2D(192, 3, strides=, activation='relu', padding='same')(y1)
    '''
    x1 = merge([x, y], mode='concat', concat_axis=CONCAT_AXIS)
    x1 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x1)
    x1 = conv2D_bn(x1, 192, 13, 13, subsample=(1, 1), border_mode='same')
    y1 = merge([x, y], mode='concat', concat_axis=CONCAT_AXIS)
    y1 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y1)
    y1 = conv2D_bn(y1, 192, 13, 13, subsample=(1, 1), border_mode='same')
    '''
    # Channel 1 - Conv Net Layer 5
    x2 = keras.layers.concatenate([x1, y1])
    x2 = Conv2D(192, 3, strides=, activation='relu', padding='same')(x2)

    # Channel 2 - Conv Net Layer 5
    y2 = keras.layers.concatenate([x1, y1])
    y2 = Conv2D(192, 3, strides=, activation='relu', padding='same')(y2)
    '''
    y2 = merge([x1, y1], mode='concat', concat_axis=CONCAT_AXIS)
    y2 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y2)
    y2 = conv2D_bn(y2, 192, 13, 13, subsample=(1, 1), border_mode='same')

    x2 = merge([x1, y1], mode='concat', concat_axis=CONCAT_AXIS)
    x2 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x2)
    x2 = conv2D_bn(x2, 192, 13, 13, subsample=(1, 1), border_mode='same')
    '''

    # Channel 1 - Cov Net Layer 6
    x3 = keras.layers.concatenate([x2, y2])
    x3 = Conv2D(128, 3, strides=, activation='relu', padding='same')(x3)

    # Channel 2 - Cov Net Layer 6
    y3 = keras.layers.concatenate([x2, y2])
    y3 = Conv2D(128, 3, strides=, activation='relu', padding='same')(y3)
    '''
    x3 = conv2D_bn(x2, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x3 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(x3)
    x3 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x3)

    y3 = conv2D_bn(y2, 128, 27, 27, subsample=(1, 1), border_mode='same')
    y3 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), dim_ordering=DIM_ORDERING)(y3)
    y3 = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(y3)
    '''

    # Channel 1 - Cov Net Layer 7
    x4 = keras.layers.multiply([x3, y3])
    x4 = Flatten()(x4)
    x4 = Dense(2048, activation='relu')(x4)
    x4 = Dropout(DROPOUT)(x4)

    # Channel 2 - Cov Net Layer 7
    y4 = keras.layers.multiply([x3, y3])
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
    x5 = keras.layers.multiply([x4, y4])
    x5 = Flatten()(x5)
    x5 = Dense(2048, activation='relu')(x5)
    x5 = Dropout(DROPOUT)(x5)

    # Channel 2 - Cov Net Layer 8
    y5 = keras.layers.multiply([x4, y4])
    y5 = Flatten()(y5)
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
    xy = keras.layers.multiply([x5, y5])
    '''
    xy = merge([x5, y5], mode='mul')
    xy = Dense(output_dim=NB_CLASS,activation='softmax')(xy)

    return xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING
    '''

def check_print():
    # Create the Model
    xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,output=[xy])
    model.summary()

    # Save a PNG of the Model Build
    #plot(model, to_file='./Model/AlexNet_Original.png')

    #model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    #print('Model Compiled')
