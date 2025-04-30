from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Flatten, Dense, Layer, BatchNormalization,
                                     Dropout, Reshape, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
from tcn import TCN
from tensorflow.keras.models import load_model
import numpy as np


def tcn_model(input_shape):
   
    from tensorflow.keras.models import Sequential
    from tcn import TCN

    model = Sequential()
    
    # TCN layer
    model.add(TCN(
        nb_filters=64,
        kernel_size=5,
        dilations=(1, 2, 4, 8, 16, 32),
        input_shape=input_shape
    ))

    # Output layer
    model.add(Dense(11, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



