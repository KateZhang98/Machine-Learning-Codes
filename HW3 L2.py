#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd
from scipy.special import expit
import sys
import math
import time

def sigmoid(x):
    return expit(x)

def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.
    # find the percentage of wrong classification?

    # Your code here, assign the proper value to test_error:
    d=X.shape[0]
    binary_error = 0
    x1=np.ones(d).reshape(d,1)
    data=np.concatenate((x1,X),axis=1)
    p = sigmoid(data@w)
    labels = []
    for i in range(len(p)):
        if(p[i] > 0.5):
            labels.append(1)
        else:
            labels.append(-1)
    labels = np.asarray(labels)
    error =0
    for i in range(len(y)):
        if (labels[i]!= y[i]):
            error+=1
    binary_error = error/len(y)

    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold,lam):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions;
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    N=X.shape[0]
    w=w_init
    d=X.shape[1]+1
    x1=np.ones(N).reshape(N,1)
    data=np.concatenate((x1,X),axis=1)
    t=0
    while t<max_its:
        #calculate direction
        sums=np.zeros(d)
        for i in range(N):
            b=1+math.exp(y[i]*data[i,:]@w)
            sums+=(y[i]*data[i,:])/b
        sums = (-sums/N).reshape(d,1)
        w=(1-2*eta*lam)*w-eta*sums
        count=0
        for j in range(d):
            if abs(sums[j])<grad_threshold:
                count+=1
#         print(count)
        if count==d:
            break
        t+=1

#     e_in =0
#     total = 0
#     for i in range(N):
#         total+= np.log(1+math. exp(-y[i]*data[i,:]@w))
#     e_in = total/N
    return t, w
# def l1(w, lam):
#     numpy.linalg.norm(w, ord=1)
def main():
    X_train, X_test, y_train, y_te = np.load("digits_preprocess.npy", allow_pickle=True)
#     print(X_train)
#     print(y_train)
    # Your code here
    d=X_train.shape[1]
    rows=X_train.shape[0]
    w_init=np.zeros(d+1).reshape(d+1,1)
    y=np.zeros(rows).reshape(rows,1)
    for i in range(rows):
        if y_train[i]==0:
            y[i]=-1
        else:
            y[i]=1
    test_row =X_test.shape[0]
    y_test = np.zeros(test_row).reshape(test_row,1)
    for i in range(test_row):
        if y_te[i]==0:
            y_test[i]=-1
        else:
            y_test[i]=1
    x_train_scale =np.zeros((rows,d))
    x_test_scale=np.zeros((test_row,d))
    for i in range(d):
        train_mean = np.mean(X_train[:,i])
        train_var=np.var(X_train[:,i])
        std=np.sqrt(train_var)
        if std !=0:
            x_train_scale[:,i]=(X_train[:,i]-train_mean)/std
            x_test_scale[:,i]=(X_test[:,i]-train_mean)/std
    lamb=[0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
#     print(x_train_scale[0])
    errates=[]
    zeros = []
    for rate in lamb:
        t,w= logistic_reg(x_train_scale, y, w_init, pow(10,4), 0.01, pow(10,-6),rate)
        t_error =find_binary_error(w,x_test_scale,y_test)
        errates.append(t_error)
        num_zero=np.count_nonzero(w==0)
        zeros.append(num_zero)

    print(errates)
    print(zeros)
#     print(w)
#     print(num_zero)

if __name__ == "__main__":
    main()
