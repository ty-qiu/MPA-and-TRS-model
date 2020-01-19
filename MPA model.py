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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import tree

from keras import models
from keras import layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import roc_curve,auc,roc_auc_score
import tensorflow as tf
from keras import backend as K
import os
from numpy.random import seed 
seed(1)  

def neural_network(x_train,y_train):
    #scaler = StandardScaler()
    #scaler.fit(x_train)
    #x_train = scaler.transform(x_train)
    #x_test = scaler.transform(x_test)
    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=5000)
    mlp.fit(x_train,y_train)
    return mlp  

def dealWithSVM(x_train,y_train):
    svc = svm.SVC(probability = True)
    svc.fit(x_train,y_train)
    return svc
    
def dealWithLR(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr
    
def dealWithNB(X, y):
    nb = GaussianNB()
    nb.fit(X, y)
    return nb
    
    
data_train=pd.read_csv("train.csv")
data_test=pd.read_csv('test.csv')

feature=[]  
for i in data_train.columns:
    if (i!='label5') & (i!='sample') & (i!='OS'):
        feature.append(i)
sample=['sample']
#sample2=['sample']
sample_train=data_train[sample]
sample_test=data_test[sample]

label=['label5']
x_train=data_train[feature] 
y_train=data_train[label]  
x_test=data_test[feature]  
y_test=data_test[label]      
x_train=x_train.as_matrix()  
y_train=y_train.as_matrix()  
x_test=x_test.as_matrix()     
y_test=y_test.as_matrix()     
sample_train=sample_train.as_matrix()
#sample_test=sample_test.as_matrix()
sample_test_OS=data_test['OS']

###(1) neural_network   
mlp=neural_network(x_train,y_train)    
#print mlp.coefs_
#print mlp.intercepts_
predictions_neural_network = mlp.predict(x_test)
probablity_neural_network= mlp.predict_proba(x_test)
fpr_neural_network,tpr_neural_network, threshold_neural_network = roc_curve(y_test,probablity_neural_network[:, 1])


###(2)LR
LR=dealWithLR(x_train,y_train)    
LR.fit(x_train,y_train)
predictions_LR = LR.predict(x_test)
probablity_LR= LR.predict_proba(x_test)
fpr_LR,tpr_LR, threshold_LR = roc_curve(y_test,probablity_LR[:, 1])


###(3)NB
NB=dealWithNB(x_train,y_train)    
NB.fit(x_train,y_train)
predictions_NB = NB.predict(x_test)
probablity_NB= NB.predict_proba(x_test)
fpr_NB,tpr_NB, threshold_NB = roc_curve(y_test,probablity_NB[:, 1])


###(4)SVM
SVM=dealWithSVM(x_train,y_train)    
SVM.fit(x_train,y_train)
predictions_SVM= SVM.predict(x_test)
probablity_SVM= SVM.predict_proba(x_test)
fpr_SVM, tpr_SVM, threshold_SVM = roc_curve(y_test,probablity_SVM[:, 1]) 

###(5)Random Forest
RF=RandomForestClassifier(n_estimators=1000, max_depth=50, min_samples_split=5,random_state=1) 
RF.fit(x_train,y_train)
predictions_RF= RF.predict(x_test)
probablity_RF= RF.predict_proba(x_test)
fpr_RF, tpr_RF, threshold_RF = roc_curve(y_test,probablity_RF[:, 1]) 

##(6) Linear Regression
LiR=LinearRegression()
LiR.fit(x_train,y_train)
predictions_LiR = LiR.predict(x_test)
#probablity_LiR= LiR.predict_proba(x_test)
fpr_LiR,tpr_LiR,threshold_LiR = roc_curve(y_test,predictions_LiR[:, 0])

###(7)Decision Tree
mode = tree.DecisionTreeClassifier(criterion='gini')
mode.fit(x_train,y_train)
predictions_tree = mode.predict(x_test)
probablity_tree= mode.predict_proba(x_test)
fpr_tree,tpr_tree,threshold_tree= roc_curve(y_test,probablity_tree[:, 1])

###(8)Deeplearn
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(105,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=4,batch_size=512)
y_pred_label = model.predict_classes(x_test)
y_pred_prob = model.predict_proba(x_test)
fpr_Keras,tpr_Keras,threshold_Keras=roc_curve(y_test,y_pred_prob,pos_label=1)


fpr=(fpr_neural_network,fpr_LR,fpr_NB,fpr_SVM,fpr_RF,fpr_LiR,fpr_tree,fpr_Keras)
tpr=(tpr_neural_network,tpr_LR,tpr_NB,tpr_SVM,tpr_RF,tpr_LiR,tpr_tree,tpr_Keras)
labels=['NN','LR','NB','SVM',"RF","Linear","Tree","Keras"] 

##ROC curve 
def roc_curve_(fpr,tpr,labels):  
    colorTable = ['blue','red','yellow','black',"green","orange","gray","purple"]  
    plt.figure()    
    lw = 2    
    plt.figure(figsize=(10,8))  
    for i in range(len(fpr)):  
        roc_auc = auc(fpr[i],tpr[i])  
        plt.plot(fpr[i],tpr[i],color=colorTable[i],linewidth=3,label='%s (%0.4f)' %(labels[i],roc_auc))     
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')    
    plt.xlim([0.0, 1.0])    
    plt.ylim([0.0, 1.05])    
    plt.xlabel('False Positive Rate', fontsize=16)    
    plt.ylabel('True Positive Rate', fontsize=16)    
    plt.title('ROC Curve', fontsize=20)    
    plt.legend(loc="lower right")    
    plt.savefig("ROC curve.jpg")
    plt.show() 
roc_curve_(fpr,tpr,labels)