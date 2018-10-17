# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:21:11 2018

@author: lizhihuan
"""

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
    
def reconstruction_weights(Y,X_tr,K):
    cls=np.unique(Y)
    W=np.zeros((Y.shape[0],Y.shape[0]))
    for c in cls:
        print(c)
        ind=np.where(Y==c)[0]
        for i in ind:
            dis=1-scipy.spatial.distance.cdist(X_tr[ind],X_tr[i][np.newaxis,:])
            ind_=ind[np.argsort(np.squeeze(dis))][1:K+1]
            subsample=X_tr[ind_].T
            coeff=scipy.linalg.inv(subsample.T.dot(subsample)).dot(subsample.T).dot(X_tr[i].T)
            coeff=np.squeeze(coeff/np.sum(coeff,axis=0))
            
            W[i][ind_]=coeff
            W[i][i]=0
           
        
        
    
        
        
    return W

def solve_NPE(X,W,K):
    X=X.T
    M=(np.eye(X.shape[1])-W).dot((np.eye(X.shape[1])-W).T)
    T1=X.dot(M).dot(X.T)
    T2=X.dot(X.T)
    T=scipy.linalg.inv(T2).dot(T1)
    eigVals,eigVects=np.linalg.eig(T)  #求特征值，特征向量
    eigValInd=np.argsort(eigVals)
    eigValInd=eigValInd[:(-K-1):-1]
    w=eigVects[:,eigValInd]
    return w

    
    
    
    
    


K=85
S_tr,Y,X_tr=load_mat(r"/home/lizhihuan/SAE/awa_demo_data.mat")
W=reconstruction_weights(Y,X_tr,100)
w=solve_NPE(S_tr,W,K)
dr_S=w.dot(S_tr.T).T
np.save('S.npy',dr_S)






