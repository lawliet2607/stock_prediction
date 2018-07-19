# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:18:51 2018

@author: L
"""

import statsmodels.api as sm
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.preprocessing import StandardScaler
import time 
import dataprocess as dp
import numpy as np
import models
    
def predict(id, t,predicted=0):
    start=time.time()
    stockwin=dp.createWinData(logr,id,t)
    tr, te = dp.splitData(stockwin, ratio = 0.9)
    tr[:,-1]=dp.transUpDown1(tr[:,-1])
    te[:,-1]=dp.transUpDown1(te[:,-1])
    
    tr1=[]
    for i in range(len(tr)):
        if tr[i,-1] >= 0:
            tr1.append(tr[i,:])
    tr1=np.array(tr1)
    te1=[]
    for i in range(len(te)):
        if te[i,-1] >= 0:
            te1.append(te[i,:])
    te1=np.array(te1)
    
    np.random.shuffle(tr1)  
    xtr=tr1[:,:-1]
    ytr=tr1[:,-1]
    xte=te1[:,:-1]
    yte=te1[:,-1]
    
    scaler = StandardScaler()
    scaler.fit(xtr)
    xtr=scaler.transform(xtr)
    xte=scaler.transform(xte)
    ''' 
    logit=sm.Logit(ytr,xtr)
    result=logit.fit()
    ypr0=result.predict(xte)
    ypr0=np.reshape(ypr0,(ypr0.shape[0],1))
    temp1, tmep2=dp.para(0.5,0.5,ypr0,yte,1)
    '''
    xtr=np.reshape(xtr,(xtr.shape[0],t,int(xtr.shape[1]/t)))
    xte=np.reshape(xte,(xte.shape[0],t,int(xte.shape[1]/t)))
    
    model=models.classifier(t,36)
    #model.summary()
    model.fit(xtr, ytr, epochs = 100, batch_size=32,verbose=0,
              validation_split = 0.1)#, callbacks = [earlystop,history])
    #history.loss_plot('epoch')
    model.evaluate(xte, yte)
    ypr=model.predict(xte)
    if predicted==0:
        print('time cost:',time.time()-start)
        return yte, ypr
    else:
        yptr=model.predict(xtr)
        a,b=dp.maxpara(yptr,ytr)
        print('alpha:',a)
        print('beta:',b)
        true,pre=dp.para(a,b,ypr,yte,1)
        print('time cost:',time.time()-start)
        return true, pre
    
    
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
    auc=(ma[0,0]+ma[1,1])/(ma[0,0]+ma[1,1]+ma[0,1]+ma[1,0])
    roc=roc_auc_score(true, prenew)
    print('confusion matrix, accurate, roc:')
    print(ma,auc,roc)
    
def softvote3(true, pre1, pre2, pre3):
    pre=(pre1+pre2+pre3)/3
    a,b=dp.maxpara(pre,true)
    print('alpha:',a)
    print('beta:',b)
    dp.para(0.5,0.5,pre,true,1)

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
    
'''
up=[]
for i in range(len(yte)):
    if yte[i] == 1:
        up.append(ypr6[i])
up=np.array(up)
down=[]
for i in range(len(yte)):
    if yte[i] == 0:
        down.append(ypr6[i])
down=np.array(down)

def profit(price,predict):
    profit=0
    T=len(price)
    for i in range(1,T):
        if predict[i]==1:
            profit+=(price[i+1]-price[i])
    return profit
'''
def result(id):
    print('id:',id)
    true,pre1=predict(id,2)
    #true,pre2=predict(id,4)
    #true,pre3=predict(id,6)
    softvote3(true,pre1,pre1,pre1)

    