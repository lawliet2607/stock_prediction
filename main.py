# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:09:35 2018

@author: L
"""

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback
import dataprocess as dp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import models
import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1024)

filename1 = 'D:/L/Studies/HK/data/kospi/px_data.csv'  # price
filename2 = 'D:/L/Studies/HK/data/kospi/rn_data.csv'  # log return

price, datep = dp.loadCSVfile(filename1)
logr, dater= dp.loadCSVfile(filename2)
# closing price
#stock_c, date_c = dp.closingPrice(stock, date)
#scaler.fit(stock_c)
#stock_c_norm = scaler.transform(stock_c)
class LossHistory(Callback):
    '''
    储存loss和auc并画训练图
    '''
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'red', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'blue', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'yellow', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'black', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()

def run(ycol = 0, method = 'gru'):
    '''
    运行主程序
    '''
    start = time.time()
    predict_win = 1
    #  数据处理
    if 'lstm' in method or 'gru' in method:
        timesteps = 1
    elif 'rnn' in method:
        timesteps = 8
    stock_win = dp.createWinData(price, ycol, timesteps, predict_win)
    train, test = dp.splitData(stock_win, ratio = 0.9)
    scaler = StandardScaler()
    scaler.fit(train)
    train_norm = scaler.transform(train)
    test_norm = scaler.transform(test)
    #last_value = train[-1, -predict_win:]  # 测试集上一个数据
    #np.random.shuffle(train_norm)  #  打乱训练数据
    #y_tr = dp.transUpDown(train_norm[:,-predict_win:], last_value)
    y_tr = train_norm[:, -1]
    x_tr = train_norm[:, :-1]
    #y_te = dp.transUpDown(test_norm[:,-predict_win:], last_value)
    y_te = test_norm[:, -1]
    x_te = test_norm[:, :-1]
    x_tr = np.reshape(x_tr, (x_tr.shape[0], timesteps, int(x_tr.shape[1] / timesteps)))
    x_te = np.reshape(x_te, (x_te.shape[0], timesteps, int(x_te.shape[1] / timesteps)))
    y_tr = np.reshape(y_tr, (y_tr.shape[0], 1))
    y_te = np.reshape(y_te, (y_te.shape[0], 1))
    layers = [x_tr.shape[0], x_tr.shape[1], x_tr.shape[2], predict_win]
    # 模型建立
    if 'lstm' in method:
        model = models.lstm_model(layers)
    elif 'rnn' in method:
        model = models.rnn_model(layers)
    elif 'gru' in method:
        model = models.gru_model(timesteps,28)        
    # 模型训练
    earlystop = EarlyStopping(monitor = 'val_loss', patience = 10, 
                                  verbose = 0, mode = 'min')    
    history = LossHistory()
    model.fit(x_tr, y_tr, epochs = 200, batch_size = 16, shuffle=True,
              verbose=1,validation_split = 0.1,callbacks = [earlystop,history])
    score = model.evaluate(x_te, y_te, batch_size = 16)
    print('test score:', score)
    #history.loss_plot('epoch')
    #  模型测试
    #y_tr_pre = model.predict_classes(x_tr, batch_size = 32)
    y_te_pre = model.predict(x_te, batch_size = 32)
    true = y_te * scaler.scale_[-1] + scaler.mean_[-1]
    predict = y_te_pre * scaler.scale_[-1] + scaler.mean_[-1]
    #y = np.concatenate((y_tr, y_te))
    #y = y.reshape(y.shape[0])
    #dp.plotCompare(y, y_tr_pre.reshape(y_tr_pre.shape[0]), 
    #               y_te_pre.reshape(y_te_pre.shape[0]), scaler)
    dp.plotTestCompare(true, predict) 
    dp.errorPercentage(true, predict)
    print('the stock:', ycol)
    print('the method:', method)
    print('timesteps:', timesteps)
    #  预测结果
    #print('train')
    #dp.plotUpDown(y_tr, y_tr_pre)
    #print('test')
    #dp.plotUpDown(y_te, y_te_pre)
    '''
    test_len = len(y_te)
    diff, diff_pre = [], []
    for i in range(test_len):
        if i == 0:
            diff.append((true[0] - last_value) * 100 / last_value)
            diff_pre.append((predict[0] - last_value) * 100 / last_value)
        else:
            diff.append((true[i] - true[i - 1]) * 100 / true[i - 1])
            diff_pre.append((predict[i] - true[i - 1]) * 100 / true[i - 1])
    plt.scatter(diff, diff_pre)
    plt.grid(True)
    plt.xlabel('diff')
    plt.ylabel('diff_pre')
    plt.title('Scatter Plot')
    plt.show()            
    diff_label = dp.transUpDown1(diff)
    diff_pre_label = dp.transUpDown1(diff_pre)
    dp.plotUpDown(diff_label, diff_pre_label)
    '''
    print('time cost:', time.time() - start)
    #return true, predict, np.array(diff), np.array(diff_pre)