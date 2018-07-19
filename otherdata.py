# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:48:13 2018

@author: L
"""

import dataprocess as dp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import models
from keras.callbacks import EarlyStopping
import statsmodels.api as sm
import time

hkprice,datehk=dp.loadCSVfile('D:/L/Studies/HK/data/HSI.csv')
jpprice,datejp=dp.loadCSVfile('D:/L/Studies/HK/data/N225.csv')
hklogr,datehkr=dp.loadCSVfile('D:/L/Studies/HK/data/HSI_r.csv')
jplogr,datejpr=dp.loadCSVfile('D:/L/Studies/HK/data/N225_r.csv')

pricedata=jpprice
timesteps=1

def priceforotherdata(pricedata, timesteps=1):
    t=timesteps
    pricewin = dp.createWinData(pricedata, 3, t)
    tr,te = dp.splitData(pricewin, ratio = 0.9)
    scaler = StandardScaler()
    scaler.fit(tr)
    tr = scaler.transform(tr)
    te = scaler.transform(te)
        
    y_tr = tr[:, -1]
    x_tr = tr[:, :-1]
    y_te = te[:, -1]
    x_te = te[:, :-1]
    x_tr = np.reshape(x_tr, (x_tr.shape[0], t, int(x_tr.shape[1]/t)))
    x_te = np.reshape(x_te, (x_te.shape[0], t, int(x_te.shape[1]/t)))
    y_tr = np.reshape(y_tr, (y_tr.shape[0], 1))
    y_te = np.reshape(y_te, (y_te.shape[0], 1))
    model = models.gru_model(t,5) 
    earlystop = EarlyStopping(monitor = 'val_loss', patience = 10, 
                                      verbose = 0, mode = 'min')    
    model.fit(x_tr, y_tr, epochs = 100, batch_size = 32, shuffle=True,
                  verbose=2,validation_split = 0.1,callbacks = [earlystop])
    score = model.evaluate(x_te, y_te, batch_size = 32)
    print('test score:', score)
    y_te_pre = model.predict(x_te, batch_size = 32)
    true = y_te * scaler.scale_[-1] + scaler.mean_[-1]
    predict = y_te_pre * scaler.scale_[-1] + scaler.mean_[-1]
    dp.plotTestCompare(true, predict) 
    dp.errorPercentage(true, predict)

def trendforotherdata(logrdata,timesteps,predicted=0):
    t=timesteps
    start=time.time()
    logrwin=dp.createWinData(logrdata[:,:-1],3,t)
    logrwin=np.column_stack((logrwin,logrdata[t:,-1]))
    tr,te=dp.splitData(logrwin,ratio=0.9)
    np.random.shuffle(tr)
    x_tr=tr[:,:-2]
    y_tr=tr[:,-1]
    x_te=te[:,:-2]
    y_te=te[:,-1]
 
    scaler = StandardScaler()
    scaler.fit(x_tr)
    x_tr=scaler.transform(x_tr)
    x_te=scaler.transform(x_te)
    '''
    logit=sm.Logit(y_tr,x_tr)
    result=logit.fit()
    y_p0=result.predict(x_te)
    y_p0=np.reshape(y_p0,(y_p0.shape[0],1))
    print(dp.para(0.5,0.5,y_p0,y_te,1))
    '''
    x_tr=np.reshape(x_tr,(x_tr.shape[0],t,int(x_tr.shape[1]/t)))
    x_te=np.reshape(x_te,(x_te.shape[0],t,int(x_te.shape[1]/t)))
    
    model=models.classifier(t,5)
    model.fit(x_tr, y_tr, epochs = 50, batch_size=32,verbose=1,
              validation_split = 0.1)#, callbacks = [earlystop,history])
    #history.loss_plot('epoch')
    model.evaluate(x_te, y_te)
    y_p=model.predict(x_te)
    if predicted==0:
        print('time cost:',time.time()-start)
        return y_te, y_p        
    else:
        y_ptr=model.predict(x_tr)
        a,b=dp.maxpara(y_ptr,y_tr)
        print('alpha:',a)
        print('beta:',b)
        true,pre=dp.para(a,b,y_p,y_te,1)
        print('time cost:',time.time()-start)
        return true,pre

def softvote3(true, pre1, pre2, pre3):
    pre=(pre1+pre2+pre3)/3
    a,b=dp.maxpara(pre,true)
    print('alpha:',a)
    print('beta:',b)
    dp.para(0.505,0.505,pre,true,1)
    
def hardvote3(true, pre1, pre2, pre3):
    ylen=len(true)
    prenew=[]
    for i in range(ylen):
        temp=pre1[i]+pre2[i]+pre3[i]
        if temp>1.5:
            prenew.append(1)
        else:
            prenew.append(0)
    ma=confusion_matrix(true,prenew,labels=(0,1))
    acc=(ma[0,0]+ma[1,1])/(ma[0,0]+ma[1,1]+ma[0,1]+ma[1,0])
    auc=roc_auc_score(true, prenew)
    print('confusion matrix, accurate, roc:')
    print(ma,acc,auc)
'''    
true,pre1=trendforotherdata(hklogr,1,1)
true,pre2=trendforotherdata(hklogr,2,1)
true,pre3=trendforotherdata(hklogr,3,1)
softvote3(true,pre1,pre2,pre3)        
hardvote3(true,pre1,pre2,pre3)
'''