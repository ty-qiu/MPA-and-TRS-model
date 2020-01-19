# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.metrics import roc_curve, auc 
from scipy import interp  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import os
from numpy.random import seed 
seed(1)   


def dealWithLR(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr
    
data_train=pd.read_csv("train.csv")
feature=[] 
for i in data_train.columns:
    if (i!='label5') & (i!='sample') & (i!='OS'):
        feature.append(i)
sample=['sample']

sample_train=data_train[sample]
label=['label5']
x_train=data_train[feature] 
y_train=data_train[label] 
x_train=x_train.as_matrix()  
y_train=y_train.as_matrix()   
sample_train=sample_train.as_matrix()



file_path='single/'   #### the Path of the simulated file
file_list=os.listdir(file_path)
for f in file_list:
    data_test=pd.read_csv(file_path+f)
    sample1=['true_negative_sample']
    sample_test=data_test[sample1]
    x_test=data_test[feature]   
    y_test=data_test[label]     
    x_test=x_test.as_matrix()    
    y_test=y_test.as_matrix()     
    sample_test_OS=data_test['OS']

    LR=dealWithLR(x_train,y_train)    
    LR.fit(x_train,y_train)
    predictions_LR = LR.predict(x_test)
    probablity_LR= LR.predict_proba(x_test)
    fpr_LR,tpr_LR, threshold_LR = roc_curve(y_test,probablity_LR[:, 1])

    predict_lable1=pd.DataFrame(predictions_LR)
    probablity=pd.DataFrame(predictions_LR)
    result1=[sample_test,sample_test_OS,predict_lable1]
    result1_new=pd.concat(result1,axis=1)  
    result1_new.to_csv('predict_lable/'+f, index=False)
    
    fpr=fpr_LR
    tpr=tpr_LR
    labels='LR'    
    def roc_curve_(fpr,tpr,labels):
       plt.figure()    
       lw = 2    
       plt.figure(figsize=(10,8))  
       roc_auc = auc(fpr,tpr)
       plt.plot(fpr,tpr,linewidth=2,label='%s ROC curve (area = %0.4f)' %(labels,roc_auc))     
       plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')    
       plt.xlim([0.0, 1.0])    
       plt.ylim([0.0, 1.05])    
       plt.xlabel('False Positive Rate', fontsize=16)    
       plt.ylabel('True Positive Rate', fontsize=16)    
       plt.title(f.split(".")[0]+'ROC Curve', fontsize=20)    
       plt.legend(loc="lower right")    
       plt.savefig("ROC/"+f.split(".")[0]+".jpg")
       plt.show() 
    roc_curve_(fpr,tpr,labels)
   