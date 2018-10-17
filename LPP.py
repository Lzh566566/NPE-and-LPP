# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:43:11 2018

@author: lizhihuan
"""

from sklearn import neighbors
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.io as scio 
import numpy as np
import scipy
def load_mat(file):
    mat_file=scio.loadmat(file)
    S_tr=mat_file['S_tr']
    Y=np.squeeze(mat_file['param']['train_labels'][0][0])
    X_tr=mat_file['X_tr']
    return S_tr,Y,X_tr
    
def creat_W(Y,X_tr):
    cls=np.unique(Y)
    W=np.zeros((Y.shape[0],Y.shape[0]),dtype=np.int32)
    for c in cls:
        print(c)
        ind=np.where(Y==c)[0]
        for i in ind:
            dis=1-scipy.spatial.distance.cdist(X_tr[ind],X_tr[i][np.newaxis,:],'cosine')
            W[i][ind]=np.squeeze(dis)
            W[i][i]=0
        
    
        
        
    return cls,W
def creat_W_1(Y,X_tr):
    cls=np.unique(Y)
    W=np.zeros((Y.shape[0],Y.shape[0]),dtype=np.int32)
    for c in cls:
        ind=np.where(Y==c)[0]
        for i in ind:
            W[i][ind]=1
            W[i][i]=0
        
    
        
        
    return cls,W

def create_D(W):
    sum_W=np.sum(W,axis=0)
    return np.diag(sum_W)

def solve_LPP(X,L,D,K):
    X=X.T
    T1=X.dot(L).dot(X.T)
    T2=X.dot(D).dot(X.T)
    T=scipy.linalg.pinv(T2).dot(T1)
    eigVals,eigVects=np.linalg.eig(T)  #求特征值，特征向量
    eigValInd=np.argsort(eigVals)
    eigValInd=eigValInd[:(-K-1):-1]
    w=eigVects[:,eigValInd]
    return w

    
    
    
    
    


K=85
S_tr,Y,X_tr=load_mat(r"/home/lizhihuan/SAE/awa_demo_data.mat")
cls,W=creat_W(Y,X_tr)
D=create_D(W)
Laplacian_matrix=D-W
w=solve_LPP(S_tr,Laplacian_matrix,D,K)
dr_S=w.dot(S_tr.T).T
np.save('S.npy',dr_S)





