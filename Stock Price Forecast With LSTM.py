# End to End Machine Learning Project 5
#--------------------------- Brief on Long Short Term Networks------------------------------#

#--------------------------- Brief on Long Short Term Networks------------------------------#


import numpy as np
import pandas as pd
import math
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Dense, Dropout, Activation,
Flatten, MaxPooling2D,SimpleRNN)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')



n_steps = 13
n_features = 1

model = Sequential()
model.add(SimpleRNN(512, activation='relu',input_shape=(n_steps, n_features),return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.compile(optimizer='rmsprop', loss='mean_squared_error',metrics=['mse'])


def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)




