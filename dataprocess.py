# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 12:51:10 2018

@author: L
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def loadCSVfile(filename):
    '''
    载入数据kospi
    '''
    temp = np.loadtxt(filename, dtype=np.str, delimiter=",")
    date = temp[1:,0]  # 加载时间标签部分
    stock = temp[1:,1:].astype(np.float)  # 加载数据部分
    return stock, date  # 返回array类型的数据
    
def outliers(data):
    '''
    检测离群点并处理
    '''
    col_num = len(data[0])
    row_num = len(data)
    Q1 = np.percentile(data, 25, axis = 0)
    Q3 = np.percentile(data, 75, axis = 0)
    max_est = Q3 + 3 * (Q3 - Q1)
    min_est = Q1 - 3 * (Q3 - Q1)
    for i in range(col_num):
        if min_est[i] < 0:
            min_est[i] = Q1[i] - 1.5 * (Q3[i] - Q1[i])
    for j in range(col_num):
        for i in range(row_num):
            if data[i,j] > max_est[j] or data[i,j] < min_est[j]:
                data[i,j] = np.nan
    df = pd.DataFrame(data)
    df = df.fillna(method = 'pad')
    data = np.array(df)
    return data

def closingPrice(stock, date):
    '''
    提取每日收盘价，共1235天
    '''
    stock_closing = []
    date_closing = []
    for i in range(len(date)):
        if '14:00' in date[i]:
            date_closing.append(date[i][:8])
            stock_closing.append(stock[i])
    return np.array(stock_closing), np.array(date_closing)
    
def describe(id,price):
   stock=price[:,id]
   mean=np.mean(stock)
   std=np.std(stock)
   n=len(stock)
   up,down,unchange=0,0,0
   for i in range(n-1):
       if stock[i+1]>stock[i]:
           up+=1
       elif stock[i+1]<stock[i]:
           down+=1
       else:
           unchange+=1
   temp=min(up,down,unchange)
   print('id:',id)
   print('mean,std,up:down:unchange')
   print(round(mean,3),round(std,3),round(up/temp,3),':',round(down/temp,3),':',round(unchange/temp,3))
    
def createWinData(data, ycol = 0, timesteps = 1, predict_window = 1):
    '''
    创造滑动窗口数据
    '''
    #np.random.seed(1024)
    datacol = len(data[0])
    if ycol < -datacol or ycol >= datacol:
        return 'y out of length of data column'
    datarow = len(data)
    if timesteps > datarow - 1:
        return 'timesteps out of length of data row'
    dataWin = data[timesteps: datarow - predict_window + 1, ycol]
    for i in range(1, predict_window):
        Y = data[timesteps + i: datarow - predict_window + i + 1, ycol]
        dataWin = np.column_stack((dataWin, Y))
    for i in range(timesteps):
        X = data[i: datarow - predict_window -timesteps + 1 + i, ]
        dataWin = np.column_stack((X, dataWin))
    return dataWin
    
def splitData(data, ratio = 0.9, timesteps = 1):
    '''
    把数据分成两部分
    '''
    train_num = round(ratio * len(data))
    #test_num = len(data) - train_num
    train = data[:train_num,]
    #validation = data[train_num - test_num :train_num,]
    test = data[train_num:,]
    return train, test
    
def transUpDown(data, y0):
    '''
    把数据转化为up/down形式：1涨；0跌或持平
    '''
    data_updown = []
    datalen = len(data)
    for i in range(datalen):
        if i == 0:
            if data[i] > y0:
                data_updown.append(1)
            else:
                data_updown.append(0)                    
        elif data[i] > data[i - 1]:
            data_updown.append(1)
        else:
            data_updown.append(0)
    return np.array(data_updown)

def transUpDown1(data):
    data_updown = []
    datalen = len(data)
    for i in range(datalen):
        if data[i] > 0:
            data_updown.append(1)
        elif data[i] < 0:
            data_updown.append(0)
        else:
            data_updown.append(-1)
    return np.array(data_updown)

def para(a,b,pre,true,show=0):
    prel,truel=[],[]
    ylen=len(pre)
    for i in range(ylen):
        if pre[i]<a:
            truel.append(true[i])
            prel.append(0)
        elif pre[i]>b:
            truel.append(true[i])
            prel.append(1)
    ma=confusion_matrix(truel,prel)
    #print(ma,a,b)
    if ma!=[]:
        target=2*(ma[0][0]+ma[1][1])-ylen
        #target=roc_auc_score(truel, prel)
    else:
        target=-ylen
    #target=(ma[0,0]+ma[1,1])-(ma[0,1]+ma[1,0])
    if show==0:
        return target
    else:
        roc=roc_auc_score(truel, prel)
        acc=(ma[0,0]+ma[1,1])/(ma[0,0]+ma[1,1]+ma[0,1]+ma[1,0])
        nodeci=ylen-(ma[0,0]+ma[0,1]+ma[1,0]+ma[1,1])
        print('confusion matrix, accurate, auc, no decision:')
        print(ma,acc,roc,nodeci)
        return true, pre

def maxpara(pre,true):
    temp={}
    for i in range(101):
        a=0.4+i/500
        for j in range(101-i):
            b=a+j/500
            c=para(a,b,pre,true)
            temp.update({(a,b):c})
    maxkey=max(temp,key=temp.get)
    return maxkey
    
def errorPercentage(y_true, y_pre):
    '''
    计算平均误差率
    '''
    ep_ave = np.mean(abs((y_true - y_pre) / y_true) * 100)
    ep_max = float(max(abs((y_true - y_pre) / y_true) * 100))
    print('max ep:', ep_max)
    print('average ep:', ep_ave)
    
#def accuracy(label_pre, label_true):
    
    
def plotCompare(stock_true, stock_predict_train, stock_predict_test, scaler):
    '''
    作对比图：训练加预测
    '''
    stock_true = stock_true * scaler.scale_[-1] + scaler.mean_[-1]
    stock_predict_train = stock_predict_train * scaler.scale_[-1] + scaler.mean_[-1]
    stock_predict_test = stock_predict_test * scaler.scale_[-1] + scaler.mean_[-1]
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    x1 = range(len(stock_predict_train))
    x2 = range(len(stock_predict_train), len(stock_predict_test) + len(stock_predict_train))
    x = range(len(stock_true))
    ax.plot(x, stock_true, label = 'true') 
    ax.plot(x1, stock_predict_train, label='train predict', color = 'green') 
    ax.plot(x2, stock_predict_test, label='test predict', color = 'red') 
    plt.xlabel('Date') 
    plt.ylabel('Stock Price') 
    plt.title('Comparison plot') 
    plt.legend(loc = 0) 
    plt.show()
    
def plotTestCompare(stock_true, stock_predict_test):
    '''
    作预测对比图
    '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    x = range(len(stock_true))
    ax.plot(x, stock_true, label = 'true', color = 'black') 
    ax.plot(x, stock_predict_test, label='predict', color = 'grey') 
    plt.xlabel('Date') 
    plt.ylabel('Stock Price') 
    plt.title('Comparison plot') 
    plt.legend(loc = 0) 
    plt.show()

def plotUpDown(y_true, y_pre):
    '''
    作混淆矩阵，auc曲线和roc值
    '''
    print('confusion matrix:') 
    print(confusion_matrix(y_true, y_pre))
    fpr, tpr, thresholds = roc_curve(y_true, y_pre, pos_label=1)
    plt.plot(fpr,tpr,linewidth=2,label="ROC")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.legend(loc = 4)
    plt.show()
    print('roc:', roc_auc_score(y_true, y_pre))