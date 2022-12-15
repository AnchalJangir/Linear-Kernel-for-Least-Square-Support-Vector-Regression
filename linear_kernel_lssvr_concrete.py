# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:11:37 2022

@author: ancha
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import accuracy_score
import sklearn

df=pd.read_excel(r"C:\Users\ancha\OneDrive\Documents\SVM\rough_data\Concrete_Data.xls")
#df= np.array(df)


cols = len(df.axes[1])
rows = len(df.axes[0])
#print(cols)
#print(rows)

#checking missing values
#print(norm_df.isnull())
mean_col=list(df.mean(axis=0))
#print(mean_col)

if df.isnull=='True':
   for i in range(cols):
       (df.iloc[:,i]).fillna(mean_col[i])
    
    
       


x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norm_df= pd.DataFrame(x_scaled)
#print(norm_df)



       
X=norm_df.iloc[:,0:cols-1]
y=norm_df.iloc[:,cols-1]
#print(len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_test=np.array(X_test)
#print(y_train)
#print(X_train)

def lin_kernel(C,X_tr,y_tr,X_te,y_te):
    D=[]
    for i in range(y_tr.shape[0]+1):
        if i==0:
            D.append(0)
        else:
            D.append(y_tr[i-1])
    D=np.array(D)


    #generating I/C matrix
    #c=matrix of order len(X_train) with 1/C entry
    I=np.identity(X_tr.shape[0])
    I_c=I/C

    #generating row and column of 1's to add in the O_new  matrix
    list1=[]
    for  i in range(X_tr.shape[0]):
        list1.append(1)
    row_1=np.array(list1).reshape(1,X_tr.shape[0])


    list2=[]
    for i in range(X_tr.shape[0]+1):
        if i==0:
            list2.append(0)
        else:
            list2.append(1)
    col_add=np.array(list2)
    col_1=np.reshape(col_add, (X_tr.shape[0]+1,1))

    K_lin=sklearn.metrics.pairwise.linear_kernel(X_tr,Y=X_tr,dense_output=True)
    #adding O+I/C
    O=np.add(K_lin,I_c)
        
        #adding row and column of 1's to the O_new  matrix
    B=np.vstack((row_1,O))
        
    B_new=np.hstack((col_1,B))
    #print(B_new.shape)
    #solving linear eq system
    Z=np.linalg.solve(B_new,D )
    #print(Z)
   
    K=sklearn.metrics.pairwise.linear_kernel(X_tr,Y=X_te,dense_output=True)
    y_pred=[]
    for k in range(y_te.shape[0]):
       y1=0
       for j in range(y_tr.shape[0]):
          y1+=(Z[j+1]*(K[j][k]))
       y1+=Z[0]
       y_pred.append(y1)
    y_pred=np.array(y_pred)
    #print(y_pre)
  
     #print(y_te)
    MSE=sklearn.metrics.mean_squared_error(y_te,y_pred)
    RMSE = math.sqrt(MSE)
    MAE=sklearn.metrics.mean_absolute_error(y_te, y_pred)
    R2=sklearn.metrics.r2_score(y_te, y_pred)
     #Acc=sklearn.metrics.accuracy_score(y_tearray, y_predarr)
    return MSE,RMSE,MAE,R2
   
X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
#slicing          
#print(type(X_TRAIN))
#print(type(X_TEST))
#print(type(y_TRAIN))
#print(type(y_TEST))
K=5
L=int(X_TRAIN.shape[0]/K)
L_arr=list(np.arange(0,X_train.shape[0],L))
L_arr=np.append(L_arr,X_train.shape[0])
#print(L_arr)

X_Train=[]
y_Train=[]
left=X_train.shape[0]-(L*K)
for i in range(K):
     if left==0:
         X_Train.append(X_TRAIN[L_arr[i]:L_arr[i+1]])
         y_Train.append(y_TRAIN[L_arr[i]:L_arr[i+1]])
     else:
         if 0<=i<K-1:
             X_Train.append(X_TRAIN[L_arr[i]:L_arr[i+1]])
             y_Train.append(y_TRAIN[L_arr[i]:L_arr[i+1]])
         elif i==K-1:
             X_Train.append(X_TRAIN[L_arr[i]:L_arr[i+2]])
             y_Train.append(y_TRAIN[L_arr[i]:L_arr[i+2]])
                
X_Train=np.array(X_Train,dtype=object)   
y_Train=np.array(y_Train,dtype=object)    
 # print(X_Train[0].shape)   
 # print(X_Train[1].shape)   
 # print(X_Train[2].shape)   
 # print(X_Train[3].shape)   
 # print(X_Train[4].shape)   
        
 #print(X_Train)
 #print(y_Train)
C=[1e-5,1e-4,1e-3,1e-2,1e-1,1e1,1e2,1e3,1e4,1e5]
 #C=[1/16,1/8,1/4,1/2,2,4,8,16]


e=[]
      
for k in range(len(C)):
    error_1=[]
    
    for i in range(K):
        Xk_t=[]
        Xk_test1=[]
        yk_t=[]
        yk_test1=[]
        for j in range(K):
            if j==i:
                Xk_test1.append(X_Train[j])
                yk_test1.append(y_Train[j])
            else:
                 Xk_t.append(X_Train[j])
                 yk_t.append(y_Train[j])
        Xk_train=np.concatenate(Xk_t)
      
        yk_train=np.concatenate(yk_t)
        Xk_test=np.concatenate(Xk_test1)
        yk_test=np.concatenate(yk_test1)
        #print(Xk_train.shape)
        #print(Xk_test.shape)
               
       #print(type(Xk_train))
       #print(type(yk_train))
               
       #print( type(Xk_test))
       #print(type(yk_test))
        MSE,RMSE,MAE,R2=lin_kernel(C[k],Xk_train,yk_train,Xk_test,yk_test)
        error_1.append(MSE)
      
    #print(error_1)
    error=np.mean(np.array(error_1))
    #print(error)
    e.append((C[k],error))
#print(e)#

len=len(C)    
min=[]
for i in range(len):
    min.append(e[i][1])
#print(min)
min_error=np.min(np.array(min))
min_e_index=min.index(min_error)
#print(min_e_index)
C=e[min_e_index][0]
print(f"optimum C:{C}")
    
    
MSE,RMSE,MAE,R2= lin_kernel(C,X_TEST,y_TEST,X_test,y_test)
print(f"MSE:{MSE}")
print(f"RMSE:{RMSE}")
print(f"MAE:{MAE}")
print(f"R2:{R2}")
         