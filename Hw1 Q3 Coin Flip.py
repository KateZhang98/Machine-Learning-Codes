import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
import random
import sys
import math

def coin_flip(N,d,num_exp):
    num_heads1=np.zeros((num_exp,))
    num_headsR=np.zeros((num_exp,))
    num_headsM=np.zeros((num_exp,))
    flips=np.zeros((N,num_exp))
    for m in range(num_exp):
        for a in range(N):
            heads=0
            for b in range(d):
                flip = random.randint(0, 1)
                if (flip == 0):
                    heads+=1
            flips[a,m]=heads/d
    num_heads1=flips[0]
    for n in range(num_exp):
        headsM = min(flips[:,n])
        num_headsM[n]=headsM
    a=random.randint(0, N)
    num_headsR=flips[a]
    return num_heads1, num_headsM, num_headsR

def main():
    num_heads1, num_headsM, num_headsR = coin_flip(1000,10,100000)
    num_exp=len(num_heads1)
    e=np.arange(0,1,0.01)
    hBounds=np.zeros((len(e),))
    for n in range(len(e)):
        hBounds[n]=2/(pow(math.e,(2*pow(e[n],2)*10)))
#     print(hBounds)
    p1 =[]
    d1 = abs(num_heads1-0.5)
    for j in e:
        count = 0
        for i in range(num_exp):
            if d1[i]-j>0:
                count+=1
        p1.append(count)
    p1=np.array(p1)/100000
    pM = []
    dM = abs(num_headsM-0.5)
    for j in e:
        count = 0
        for i in range(num_exp):
            if dM[i]-j>0:
                count+=1
        pM.append(count)
    pM=np.array(pM)/100000
    pR = []
    dR = abs(num_headsR-0.5)
    for j in e:
        count = 0
        for i in range(num_exp):
            if dR[i]-j>0:
                count+=1
        pR.append(count)
    pR=np.array(pR)/100000

    plt.plot(e,p1,label='first coin')
    plt.plot(e,hBounds,label='Hoeffding Bound')
    plt.title('C1')
    plt.xlabel('epsilon')
    plt.ylabel('prbability')
    plt.legend()
    plt.show()

    plt.plot(e,pM,label='min coin')
    plt.plot(e,hBounds,label='Hoeffding Bound')
    plt.title('C2')
    plt.xlabel('epsilon')
    plt.ylabel('prbability')
    plt.legend()
    plt.show()

    plt.plot(e,pR,label='random coin')
    plt.plot(e,hBounds,label='Hoeffding Bound')
    plt.title('C3')
    plt.xlabel('epsilon')
    plt.ylabel('prbability')
    plt.legend()
    plt.show()
    plt.hist(num_heads1)
    plt.title("Fraction of heads First Coin")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fraction")
    plt.show()
    plt.hist(num_headsM)
    plt.title("Fraction of heads Min Coin")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fraction")
    plt.show()
    plt.hist(num_headsR)
    plt.title("Fraction of heads Random Coin")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fraction")
    plt.show()

if __name__ == "__main__":
    main()
