# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:13:53 2018

@author: L
"""

import numpy as np
import dataprocess as dp
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler

def ar(id):
    order=6
    train, test = dp.splitData(price[:,id], ratio = 0.9)
    #plt.plot(test)
    scaler = StandardScaler()
    scaler.fit(train)
    train_norm = scaler.transform(train)
    test_norm = scaler.transform(test)
    norm=np.append(train_norm,test_norm)

    model=ARMA(train_norm,order=(order,0))
    result=model.fit()
    a=result.params
    print(a)
    pre=[]
    s=len(train)
    for i in range(len(test)):
        pre.append(a[0]+a[1]*norm[s-1+i]+a[2]*norm[s-2+i]+a[3]*norm[s-3+i]\
        +a[4]*norm[s-4+i]+a[5]*norm[s-5+i]+a[6]*norm[s-6+i])
    print('mae:',mae(test_norm,pre))
    pre=list(map(lambda x:x*scaler.scale_+scaler.mean_, pre))
    dp.plotTestCompare(test, pre)
    dp.errorPercentage(test, pre)

def st1(id):
    train, test = dp.splitData(price[:,id], ratio = 0.9)
    scaler = StandardScaler()
    scaler.fit(train)
    train_norm = scaler.transform(train)
    test_norm = scaler.transform(test)
    norm=np.append(train_norm,test_norm)
    pre=norm[len(train)-1:-1]
    print('mae:',mae(test_norm,pre))
    dp.plotTestCompare(test_norm, pre)
    pre=pre*scaler.scale_+scaler.mean_
    dp.errorPercentage(test, pre)    