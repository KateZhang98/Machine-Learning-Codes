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


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
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
    binary_error = 0
    d=X.shape[1]+1
    x1=np.ones(N).reshape(N,1)
    data=np.concatenate((x1,X),axis=1)
    t=0
    while t<max_its:
        #calculate direction
        sum=np.zeros(d)
        for i in range(N):
            b=1+math.exp(y[i]*data[i,:]@w)
            sum+=(y[i]*data[i,:])/b
        sum = (-sum/N).reshape(d,1)
#         print(sum)
        w=w-eta*sum
        count=0
        for j in range(d):
            if abs(sum[j])<grad_threshold:
                count+=1
#         print(count)
        if count==d:
            break
        t+=1

    e_in =0
    total = 0
    for i in range(N):
        total+= np.log(1+math.exp(-y[i]*data[i,:]@w))
    e_in = total/N
#     print(w)
    return t, w, e_in


def main():
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')
#     print(train_data)
    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')

    # Your code here
    train = train_data.iloc[:,:-1]
    X=train.to_numpy()
    d=X.shape[1]
    rows=train.shape[0]
    w_init=np.zeros(d+1).reshape(d+1,1)
    y=np.zeros(rows).reshape(rows,1)
    y_init= train_data.iloc[:,-1]
    for i in range(rows):
        if y_init[i]==0:
            y[i]=-1
        else:
            y[i]=1
    test=test_data.iloc[:,:-1]
    x_test = test.to_numpy()
    test_row =x_test.shape[0]
    y_t =test_data.iloc[:,-1]
    y_test = np.zeros(test_row).reshape(test_row,1)
    for i in range(test_row):
        if y_t[i]==0:
            y_test[i]=-1
        else:
            y_test[i]=1
    x_train_scale =np.zeros((rows,d))
    x_test_scale = np.zeros((test_row,d))
    train_means=[]
    train_variance=[]
    test_means=[]
    test_variance=[]
    for i in range(d):
        train_mean = np.mean(X[:,i])
        train_var=np.var(X[:,i])
        train_means.append(train_mean)
        train_variance.append(train_var)
        test_mean=np.mean(x_test[:,i])
        test_var=np.var(x_test[:,i])
        test_means.append(test_mean)
        test_variance.append(test_var)
        x_train_scale[:,i]=(X[:,i]-train_mean)/train_var
        x_test_scale[:,i]=(x_test[:,i]-test_mean)/test_var
#     print(x_train_scale)
#     print(x_test_scale)
    train_means=np.array(train_means)
    train_variance=np.array(train_variance)
    test_means=np.array(test_means)
    test_variance=np.array(test_variance)
#     print(np.around(train_means,2))
#     print(np.around(train_variance,2))
#     print(np.around(test_means,2))
#     print(np.around(test_variance,2))
    learn_rate=[0.01, 0.1, 1, 4, 7, 7.5, 7.6, 7.7]
    ts=[]
    e_ins=[]
    errates=[]
    times=[]
    for rate in learn_rate:
        start = time.time()
        t,w,e_in = logistic_reg(x_train_scale, y, w_init, pow(10,6), pow(10,-5), pow(10,-3))
        end = time.time()
        ts.append(t)
        e_ins.append(e_in)
        times.append(end-start)
        eror = find_binary_error(w, x_train_scale, y)
        t_error =find_binary_error(w,x_test_scale,y_test)
        error=[eror,t_error]
        errates.append(error)
    print(times)
    print(ts)
    print(e_ins)
    print(errates)
if __name__ == "__main__":
    main()
