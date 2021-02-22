from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import DepthwiseConv2D
from keras import backend as K
from keras.optimizers import *
from keras.layers import *        
import tensorflow as tf


#import segmentation_models as sm

import metrics as M
import losses as L
import kernels as Kr



kernel_size = 3
sigma = 10


def get_gauss_weights(layer, kernel_size=3, sigma=5):
    "HPF weights"
    
    # Get number of channels in feature
    in_channels = layer.shape[-1]
    
    # Get kernel
    #w = Kr.gauss_2D(shape=(kernel_size, kernel_size),sigma=sigma)
    # OR
    w_blur, w_outline, w_sharpen = Kr.get_kernel()
    
    # Weights according to kernel type
    w = w_outline # w_outline aka high pass filter works well!
    
    # Change dimension
    w = np.expand_dims(w, axis=-1)
    # Repeat filter by in_channels times to get (H, W, in_channels)
    w = np.repeat(w, in_channels, axis=-1)
    # Expand dimension
    w = np.expand_dims(w, axis=-1)
    return w


# From https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def unet(input_size = (256,256,1)):
    "Baseline Unet"
    
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # Binary segmentation
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9) 
    # Multiclass segmentation
    #conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    # Compile model with optim and loss
    optim = 'adam' 
    
    # If bin seg, use bce loss, or categorical_crossentropy for multi class
    loss_func = 'binary_crossentropy'  
    
    model.compile(optimizer = optim, loss = loss_func, metrics = [M.jacard, M.dice_coef])

    return model



def wide_unet(input_size = (256,256,1)):
    "Wide Unet"
    
    inputs = Input(input_size)
    conv1 = Conv2D(35, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(35, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(70, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(70, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(140, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(140, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(280, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(280, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(560, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(560, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(280, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(280, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(140, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(140, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(70, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(70, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(35, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(35, (3, 3), activation='relu', padding='same')(conv9)

    # Binary segmentation
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9) 
    # Multiclass segmentation
    #conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    # Compile model with optim and loss
    optim = 'adam' 
    
    # If bin seg, use bce loss, or categorical_crossentropy for multi class
    loss_func = 'binary_crossentropy'
    
    model.compile(optimizer = optim, loss = loss_func, metrics = [M.jacard, M.dice_coef])

    return model




def edgeunet(input_size = (256,256,1)):
    "Unet with HPF layer in skip connections"
    
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    
    # 1. Get gaussisan weights(1, H, W, channels) 
    W = get_gauss_weights(conv4, kernel_size, sigma) 
    # 2. Build gauss layer with random weights
    gauss_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    # 3. Pass input to gauss layer
    conv4 = gauss_layer(conv4)
    # 4. Set gauss filtersas layer weights 
    gauss_layer.set_weights([W])
    # 5. Dont update weights
    gauss_layer.trainable = False 
    print(gauss_layer.get_weights()[0].shape)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    
    W = get_gauss_weights(conv3, kernel_size, sigma) 
    gauss_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv3 = gauss_layer(conv3)
    gauss_layer.set_weights([W])
    gauss_layer.trainable = False
   
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)


    W = get_gauss_weights(conv2, kernel_size, sigma) 
    gauss_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv2 = gauss_layer(conv2)
    gauss_layer.set_weights([W])
    gauss_layer.trainable = False
    
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    
    
    
    W = get_gauss_weights(conv1, kernel_size, sigma)
    gauss_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv1 = gauss_layer(conv1)
    gauss_layer.set_weights([W])
    gauss_layer.trainable = False
    
    
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # Binary segmentation
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9) 
    # Multiclass segmentation
    #conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9) 
    
    model = Model(inputs=[inputs], outputs=[conv10])
    
    # Compile model with optim and loss
    optim = 'adam' 
    
    # If binary seg, use bce loss, categorical_crossentropy if multiclass seg
    loss_func = 'binary_crossentropy' 
    
    model.compile(optimizer = optim, loss = loss_func, metrics = [M.jacard, M.dice_coef])

    return model



def wide_edgeunet(input_size = (256,256,1)):
    "Wide Unet with HPF in skip connections"
    
    inputs = Input(input_size)
    conv1 = Conv2D(35, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(35, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(70, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(70, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(140, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(140, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(280, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(280, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(560, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(560, (3, 3), activation='relu', padding='same')(conv5)
    
    
    # 1. Get gaussisan weights(1, H, W, channels) 
    W = get_gauss_weights(conv4, kernel_size, sigma) 
    # 2. Build gauss layer with random weights
    gauss_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    # 3. Pass input to gauss layer
    conv4 = gauss_layer(conv4)
    # 4. Set gauss filtersas layer weights 
    gauss_layer.set_weights([W])
    # 5. Dont update weights
    gauss_layer.trainable = False 
    print(gauss_layer.get_weights()[0].shape)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(280, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(280, (3, 3), activation='relu', padding='same')(conv6)
    
    
    W = get_gauss_weights(conv3, kernel_size, sigma) 
    gauss_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv3 = gauss_layer(conv3)
    gauss_layer.set_weights([W])
    gauss_layer.trainable = False
   
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(140, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(140, (3, 3), activation='relu', padding='same')(conv7)


    W = get_gauss_weights(conv2, kernel_size, sigma) 
    gauss_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv2 = gauss_layer(conv2)
    gauss_layer.set_weights([W])
    gauss_layer.trainable = False
    
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(70, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(70, (3, 3), activation='relu', padding='same')(conv8)
    
    
    
    W = get_gauss_weights(conv1, kernel_size, sigma)
    gauss_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv1 = gauss_layer(conv1)
    gauss_layer.set_weights([W])
    gauss_layer.trainable = False
    
    
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(35, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(35, (3, 3), activation='relu', padding='same')(conv9)

    # Binary segmentation
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9) 
    # Multiclass segmentation
    #conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9) 
    
    model = Model(inputs=[inputs], outputs=[conv10])
    
    # Compile model with optim and loss
    optim = 'adam' 
    
    # If binary seg, use bce loss, categorical_crossentropy if multiclass seg
    loss_func = 'binary_crossentropy'
    
    model.compile(optimizer = optim, loss = loss_func, metrics = [M.jacard, M.dice_coef])

    return model



from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model



def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, n_channels):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model


# UNET and EDGEUNET WITH BACKBONES

def unet_backbone(backbone, input_size, encoder_weights=None):
    
    model = sm.Unet(backbone_name=backbone, input_shape=input_size, classes=1, activation='sigmoid', encoder_weights=encoder_weights)
    
    # Compile model with optim and loss
    optim = 'adam' 
    
    # If bin seg, use bce loss, or categorical_crossentropy for multi class
    loss_func = 'binary_crossentropy'  
    
    model.compile(optimizer = optim, loss = loss_func, metrics = [M.jacard, M.dice_coef])
    
    return model



def edgeunet_backbone(backbone, input_size, encoder_weights=None):
    
    model = sm.EdgeUnet(backbone_name=backbone, input_shape=input_size, classes=1, activation='sigmoid', encoder_weights=encoder_weights)
    
    # Compile model with optim and loss
    optim = 'adam' 
    
    # If bin seg, use bce loss, or categorical_crossentropy for multi class
    loss_func = 'binary_crossentropy'  
    
    model.compile(optimizer = optim, loss = loss_func, metrics = [M.jacard, M.dice_coef])
    
    return model


