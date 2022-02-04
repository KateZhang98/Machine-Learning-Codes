#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import math


def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees using a random subset of the features
    # at each split on the input dataset and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function
    clf=DecisionTreeClassifier(criterion="entropy")
    N=len(X_train)
    M = len(X_test)
    test = np.empty((N,num_bags))
    test[:] = np.NaN
    bag_error_count=0
    test_count=0
    X_train=np.array(X_train)
    test_2=np.zeros((M,num_bags))
    np.random.seed(153)
    for i in range(num_bags):
        # N random choices of indices
        d=np.random.choice(np.arange(0,N),N)
        set_difference = set(np.arange(0,N)) - set(d)
        #test indices
        list_difference = list(set_difference)
        d = list(set(d))
        data=[]
        y=[]
        for j in range(len(d)):
            a=d[j]
            data.append(X_train[a,:])
            y.append(y_train[a])
        clf = clf.fit(data,y)
        for j in range(len(list_difference)):
            b=list_difference[j]
            test[b,i] = clf.predict(X_train[b,:].reshape(1, -1))
        test_2[:,i]=clf.predict(X_test)
    for m in range(N):
        y_pred=mode(test[m,:],nan_policy='omit')
        if y_pred[0] != y_train[m]:
            bag_error_count+=1
    for n in range(M):
        y_pred=mode(test_2[n,:])
        if y_pred[0]!=y_test[n]:
            test_count+=1
    out_of_bag_error =bag_error_count/N
#     print(out_of_bag_error)
    test_error =test_count/M
#     print(test_error)

    return out_of_bag_error, test_error
def single_decision_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_train,y_train)
    test_error = 1-clf.score(X_test,y_test)
    return test_error

def main_hw4():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')
#     print(og_train_data)
    #1-200 number of bags
    num_bags =np.arange(1,201)
    OOB =[]
    # Split data
    X_train =[]
    y_train=[]

#     print(og_train_data.shape)
    for i in range(len(og_train_data)):
        if og_train_data[i,0]==5:
            X_train.append(og_train_data[i,1:])
            y_train.append(5)
        if og_train_data[i,0]==3:
            X_train.append(og_train_data[i,1:])
            y_train.append(3)
    X_test =[]
    y_test =[]
    for i in range(len(og_test_data)):
        if og_test_data[i,0]==5:
            X_test.append(og_test_data[i,1:])
            y_test.append(5)
        if og_test_data[i,0]==3:
            X_test.append(og_test_data[i,1:])
            y_test.append(3)
#     # Run bagged trees
    for i in num_bags:
        out_of_bag_error,test_error = bagged_trees(X_train, y_train, X_test, y_test, i)
        OOB.append(out_of_bag_error)
#     print(out_of_bag_error)
#     print(test_error)
    plt.title('3 verse 5')
    plt.xlabel('number of bags')
    plt.ylabel('out of bag error')
    plt.plot(num_bags,OOB, 'r')
#     t_error = single_decision_tree(X_train, y_train, X_test, y_test)
#     print(t_error)


if __name__ == "__main__":
    main_hw4()
