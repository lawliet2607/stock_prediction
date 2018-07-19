# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:20:20 2018

@author: L
"""

from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
import numpy as np
from numpy import newaxis
        
def lstm_model(t):
    '''
    two lstm layers
    '''
    sgd = optimizers.SGD(lr = 0.01, momentum = 0.8, decay = 0.001)
    rmsprop = optimizers.RMSprop(lr = 0.01, decay = 0.001)
    model = Sequential()
    model.add(BatchNormalization(input_shape = (t, 36)))
    model.add(Conv1D(64, 1, activation='relu',
                        kernel_regularizer = regularizers.l1_l2(0.01, 0.01)))
    model.add(Conv1D(32, 1, activation='relu'))    
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(1))
    model.add(LSTM(units = 16, return_sequences = True, kernel_regularizer = regularizers.l1_l2(0.01, 0.01)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 8))
    model.add(Dropout(0.2))
    #model.add(Dense(4))
    #model.add(Activation('linear'))     
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))   
    model.summary()
    model.compile(loss = 'binary_crossentropy', 
                  optimizer = rmsprop, 
                  metrics=['accuracy'])
    return model
    
def gru_model(timesteps,features):
    '''
    two gru layers
    '''
    sgd = optimizers.SGD(lr = 0.005, momentum = 0.8, decay = 0.001)
    rmsprop = optimizers.RMSprop(lr = 0.01)
    model = Sequential()
    #model.add(BatchNormalization(input_shape=(timesteps, features)))
    model.add(GRU(units = 128, input_shape=(timesteps, features),
                  #kernel_initializer='glorot_normal',
                  return_sequences = True, 
                  kernel_regularizer = regularizers.l1_l2(0.01,0.01)))
    model.add(Dropout(0.2))
    #model.add(GRU(units = 128, return_sequences = True))
    #model.add(Dropout(0.2))
    model.add(GRU(units = 128))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    #model.add(BatchNormalization())
    model.add(Activation('linear'))
    #model.summary()
    model.compile(loss = 'mae', 
                  optimizer = sgd)
    return model
    
def rnn_model(layers):
    '''
    simple rnn layers
    '''
    sgd = optimizers.SGD(lr = 0.01, momentum = 0.8, decay = 0.001)
    model = Sequential()
    model.add(SimpleRNN(units = 128, input_shape = (layers[1], layers[2]), 
            return_sequences = False, kernel_regularizer = regularizers.l1_l2(0.01, 0.01)))
    model.add(Dropout(0.2))
    #model.add(SimpleRNN(units = 128, return_sequences = True))
    #model.add(Dropout(0.2))
    #model.add(SimpleRNN(units = 128))
    #model.add(Dropout(0.2))    
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.summary()
    model.compile(loss = 'mae', optimizer = sgd, metrics=['accuracy'])
    return model

def classifier(t,f):
    rmsprop = optimizers.RMSprop(lr = 0.01,decay=0.01)
    sgd = optimizers.SGD(lr = 0.01, momentum = 0.8, decay = 0.001)
    model = Sequential()
    #model.add(BatchNormalization(input_shape=(t, 36)))
    model.add(Dropout(0.2, input_shape=(t, f)))
    model.add(Conv1D(32, 1, activation='relu', 
              kernel_regularizer = regularizers.l1_l2(0.01,0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(1))
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])    
    return model
    
def predict(model, x_test):
    '''
    点预测
    '''
    y_pre = model.predict(x_test)    
    return y_pre
    
def predict_sequence(model, x_test, pred_len = 71):
    '''
    连续预测一天的序列
    '''
    y_pre = []
    for i in range(pred_len):
        if i == 0:
            y_pre.append(model.predict(x_test[0][newaxis]))
        else:
            y_pre.append(model.predict(np.insert(x_test[1,:,:-1][newaxis],
                                         35, y_pre[i - 1], axis = 2)))
    return np.array(y_pre)