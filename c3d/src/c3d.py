import skvideo.io
import keras.backend as K
from keras.models import Sequential
from keras.utils.data_utils import get_file
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D

WEIGHTS_PATH = 'c3d/model/sports1M_weights_tf.h5'

def C3D():
    """
    Instantiates a C3D  model
    """
    
    if K.image_data_format() == 'channels_last':
        shape = (16,112,112,3)
    else:
        shape = (3,16,112,112)
        
    model = Sequential()
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0,1,1)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    model.load_weights(WEIGHTS_PATH)

    return model
