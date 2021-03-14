# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:36:05 2019

@author: hurenlaker
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss
from scipy.stats import norm
from numpy.linalg import norm
from numpy.linalg import inv
import random
import time
import csv

###########################Problem 1
#read csv.file
data = pd.read_csv("D:\\pc2\\2020\\STA6160-statistical computing\\charlie.csv") 
data1=data[['Data','x1','x2','x3','x4']] #only choose the specific columns
rown=data1.shape[0]
colum=data1.shape[1]
ct=len(data1[data1.iloc[:,0]=='Original']) # the number of original class samples
#separate data into datax and datay
datax=data1[['x1','x2','x3','x4']]  #only contain x1,x2,x3,x4 independent variables
datay=data1[['Data']]               #only contain y dependent variable
#define gaussian kernel function 
def k(xi,xj,sigma): #xi,xj,sigma are three inputs
    kij=np.exp(-np.power(norm(xi-xj),2)/np.power(sigma,2)) #norm(xi,xj) is the L2_norm of xi,xj
    return(kij)
    
#problem 1 question a: compute the inverse of Hn

def Hn_inv(hn_1_inv,data,xn): # three inputs: the inverse of Hn-1,(n-1)iteration data,new sample Xn
    d=len(data) # the number of observation
    delta_n=np.zeros((d,1)) # column vector delta_n={k(xn,xi)}
    for i in range(d):
        delta_n[i][0]=k(xn,data[i,:],sigma)
    delta_nn=k(xn,xn,sigma)# the scalar delta_nn by xn itself
    an=np.array(np.matrix(hn_1_inv)*np.matrix(delta_n)) # (n-1)*1 vector an
    rn=delta_nn+(1/(2*C))-np.array(np.matrix(delta_n.T)*np.matrix(an))
    hn_inv=(1/rn)*np.block([[rn*hn_1_inv+np.array(np.matrix(an)*np.matrix(an.T)),-an],
            [-an.T,1]])
    return(hn_inv)
#test Hn_inv function by the inverse of H1 and H2    
C=0.01  # the penalty factor
sigma=10 # the parameter of Gaussian Kernel
h1=np.matrix(k(np.array(datax)[0],np.array(datax)[0],sigma)+1/(2*C))
h1_inv=np.array(inv(h1)) # the inverse of H1
h2_inv=Hn_inv(h1_inv,np.array(datax)[0:1],np.array(datax)[1]) # the inverse of H2
#a loop for computing the inverse of  Hn
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
n=4                # assum the number of observation=4
for i in range(n):
    print("n=",(i+1),",","inverse","of","H",(i+1),"=")
    print(hn_1_inv)
    hn_inv=Hn_inv(hn_1_inv,np.array(datax)[0:i+1],np.array(datax)[i+1])
    hn_1_inv=hn_inv
    
#problem 1 question b: compute the updated alpha1 in iterative fashion

#define a function to compute alpha1 vector
def alpha1(hn_inv,data): # two inputs: the inverse of Hn and n observations
    d=len(data)   # the number of observation
    e=np.ones((d,1)) # n*1 column vector
    kj=np.zeros((d,1)) # kn vector
    for i in range(d):
        kj[i][0]=k(data[i,:],data[i,:],sigma)
    p1=2-np.matrix(e.T)*np.matrix(hn_inv)*np.matrix(kj)
    p2=np.matrix(e.T)*np.matrix(hn_inv)*np.matrix(e)
    alpha=0.5*np.matrix(hn_inv)*(np.matrix(kj)+(p1[0,0]/p2[0,0])*np.matrix(e))
    return (alpha)
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=4  # assume we have 4 observations
for i in range(N):
    alpha_n=alpha1(hn_1_inv,np.array(datax)[0:i+1])
    hn_inv=Hn_inv(hn_1_inv,np.array(datax)[0:i+1],np.array(datax)[i+1])
    hn_1_inv=hn_inv
    print("n=",(i+1),",","alpha",(i+1),"=")
    print(alpha_n)

#problem 1 question c: compare alpha_n in iterative fashion and in direct approach

# iterative method
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=7  # assume we have 7 observations
for i in range(N):
    alpha_n=alpha1(hn_1_inv,np.array(datax)[0:i+1])
    hn_inv=Hn_inv(hn_1_inv,np.array(datax)[0:i+1],np.array(datax)[i+1])
    hn_1_inv=hn_inv
    print("n=",(i+1),",","alpha",(i+1),"=")
    print(alpha_n)
# direct approach
def Hn(data): #define a function of computing Hn directly
    d=len(data) # the number of observation
    Km=np.zeros((d,d)) # K matrix named as Km
    I=np.eye(d) # d dimensions identity matrix
    for i in range(d):
        for j in range(d):
            Km[i][j]=k(data[i,:],data[j,:],sigma)
    hn=Km+(1/(2*C))*I # compute the matrix Hn-1   
    return (hn)

hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=7  # assume we have 7 observations
for i in range(N):
    hn_inv=inv(Hn(np.array(datax)[0:i+1]))
    alpha_n=alpha1(hn_inv,np.array(datax)[0:i+1])
    print("n=",(i+1),",","alpha",(i+1),"=")
    print(alpha_n)    

#problem 1 question d: iteratively update the radius R2 and dz

# compute R2 from the training dataset    
def R2(alpha_n,train): # R2 consists of three average parts:ap1,ap2,ap3 
    ap1=0
    ap2=0
    ap3=0
    for i in range(len(train)):
        ap1+=k(train[i,:],train[i,:],sigma)  
        
    for i in range(len(train)):
        for j in range(len(train)):
            ap2+=2*alpha_n[j,0]*k(train[i,:],train[j,:],sigma) 
            
    for i in range(len(train)):
        for j in range(len(train)):
            ap3+=alpha_n[i,0]*alpha_n[j,0]*k(train[i,:],train[j,:],sigma) 
    r2=(ap1-ap2)/len(train)+ap3
    return(r2)

#compute dz of a test dataset
def DZ(alpha_n,train,test): # assume dz consists of three parts:p1,p2,p3
    p1=k(test,test,sigma)
    p2=0
    p3=0
    for i in range(len(train)):
        p2+=2*alpha_n[i,0]*k(test,train[i,:],sigma)
    for i in range(len(train)):
        for j in range(len(train)):
            p3+=alpha_n[i,0]*alpha_n[j,0]*k(train[i,:],train[j,:],sigma) 
    dz=p1-p2+p3
    return(dz)

## assume we have a train set with 7 observations and a test set with 7 observations
tn1=7
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
for i in range(tn1): 
    alpha_n=alpha1(hn_1_inv,np.array(datax)[0:i+1])
    r2=R2(alpha_n,np.array(datax)[0:i+1])
    dz=DZ(alpha_n,np.array(datax)[0:i+1],np.array(datax)[i+1])
    hn_inv=Hn_inv(hn_1_inv,np.array(datax)[0:i+1],np.array(datax)[i+1])
    hn_1_inv=hn_inv
    print("n=",(i+1))
    print("R2=",r2)    
    print("dz=",dz)
#check the next 7 rows
tn2=14
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
for i in range(tn2): 
    alpha_n=alpha1(hn_1_inv,np.array(datax)[0:i+1]) # iterative alpha_n
    r2=R2(alpha_n,np.array(datax)[0:i+1])           # iterative R2 (n)
    dz=DZ(alpha_n,np.array(datax)[0:i+1],np.array(datax)[i+1]) # dz of test vector xn
    hn_inv=Hn_inv(hn_1_inv,np.array(datax)[0:i+1],np.array(datax)[i+1]) #iterative inv(Hn)
    hn_1_inv=hn_inv
    if i>6:   #predict the next 7 rows
        print("n=",(i+1))
        print("R2=",r2)    
        print("dz=",dz)
        if dz<=r2:
            print("It's a target")
        else:
            print("It's an outlier")

#problem 1 question e: compare the training times (in sec) between the 2 algorithms

# iterative method
start_time1 = time.clock()
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=30  # assume we have 30 observations
t1=np.zeros(N)              # record the timing as n increases for iterative method
for i in range(N):
    alpha_n=alpha1(hn_1_inv,np.array(datax)[0:i+1])
    hn_inv=Hn_inv(hn_1_inv,np.array(datax)[0:i+1],np.array(datax)[i+1])
    hn_1_inv=hn_inv
    print("n=",(i+1),",","alpha",(i+1),"=")
    print("iterative method timing:",time.clock()-start_time1, "seconds")
    t1[i]=time.clock()-start_time1    

# direct approach
start_time2 = time.time()
N=30  # assume we have 30 observations
t2=np.zeros(N)    # record the timing as n increases for direct method
for i in range(N):
    hn_inv=inv(Hn(np.array(datax)[0:i+1]))
    alpha_n=alpha1(hn_inv,np.array(datax)[0:i+1])
    print("n=",(i+1),",","alpha",(i+1),"=") 
    print("direct approach timing:",time.time()-start_time2,"seconds")
    t2[i]=time.time()-start_time2

#compare the runtimes of two method
plt.plot(np.arange(1,30,1), t1, 'r--', np.arange(1,30,1), t2,'g^')
plt.xlabel('the number of iteration')
plt.ylabel('Runtime(sec)')
plt.title('Rumtimes of iterative and direct method')
plt.text(5, 1.5, 'red:  iterative method')
plt.text(5, 1.3, 'green:direct method')
plt.show()

###########################Problem 2
#problem 2 question a: obtain the block update inverse matrix Hn+k

#read the datafile
with open("E:\\Desk\\2019\\STA6160-statistical computing\\tic-tac-toe.txt") as f:
    group1 = f.read().splitlines() 
group1_data=np.array(list(csv.reader(group1, delimiter=',')))
group1_data.shape #(958, 10)

# transform categorical variables into dummy variables
for i in range(group1_data.shape[0]):
    for j in range(group1_data.shape[1]):
        if group1_data[i,j]=='x':
            group1_data[i,j]=1
        if group1_data[i,j]=='o':
            group1_data[i,j]=2
        if group1_data[i,j]=='b':
            group1_data[i,j]=3

# separate into positive and negative datasets               
data_pos=group1_data[group1_data[:,9]=='positive']  # 626 samples
data_neg=group1_data[group1_data[:,9]=='negative']  # 332 samples

# transform data type into continuous 
data_pos_x=np.zeros((data_pos.shape[0],data_pos.shape[1]-1))
for i in range(data_pos.shape[0]):#only contain independent variables
    for j in range(data_pos.shape[1]-1):           
       data_pos_x[i,j]= int(data_pos[i,j]) 
data_neg_x=np.zeros((data_neg.shape[0],data_neg.shape[1]-1)) 
for i in range(data_neg.shape[0]):#only contain independent variables
    for j in range(data_neg.shape[1]-1):           
       data_neg_x[i,j]= int(data_neg[i,j])   

#define a function to update the block inverse matrix
def HNK_inv2(hn_inv,data,newdata):# input Hn, n observations and new K observations 
    N=len(data)
    K=len(newdata)
    B=np.zeros((K,N))  # matrix B: K*N
    D=np.zeros((K,K))  # matrix D: K*K
    I=np.eye(K)
    for i in range(K):
        for j in range(N):
            B[i,j]=k(newdata[i,:],data[j,:],sigma)
    for i in range(K):
        for j in range(K):
            D[i,j]=k(newdata[i,:],newdata[j,:],sigma)
    D=D+(1/(2*C))*I
    B=np.matrix(B)
    D=np.matrix(D)
    hn_inv=np.matrix(hn_inv)
    w1=np.array(hn_inv-hn_inv*B.T*inv(-D+B*hn_inv*B.T)*B*hn_inv)
    w2=np.array(hn_inv*B.T*inv(B*hn_inv*B.T-D))
    w3=np.array(inv(B*hn_inv*B.T-D)*B*hn_inv)
    w4=np.array(inv(D-B*hn_inv*B.T))
    hnk_inv=np.block([[w1,w2],[w3,w4]])    
    return(hnk_inv)

# the results of the inverse of Hn from n=1 to n=4 are same as before 
    
#problem 2 question b: check Hn+k  using the direct method

#if K=1, the results should be same as the inverse of Hn by Hn-1 in problem1
C=5  # the penalty factor
sigma=1 # the parameter of Gaussian Kernel
h1=np.matrix(k(data_pos_x[0],data_pos_x[0],sigma)+1/(2*C))
h1_inv=np.array(inv(h1)) # the inverse of H1

#the iterative block method
hn_inv=h1_inv    # the inverse of Hn starts from the inverse of H1
n=5
#a loop for computing the inverse of  Hn+1, K=1
for i in range(n): # compute Hn by the iterative block method, K=1
    print("K=1,n=",(i+1),",","inverse","of","H",(i+1),"=")
    print(hn_inv)
    hnk_inv=HNK_inv2(hn_inv,data_pos_x[0:i+1],data_pos_x[i+1:i+2])
    hn_inv=hnk_inv

for i in range(n): # compute alpha_n by the iterative block method, K=1
    alpha_n=alpha1(hn_inv,data_pos_x[0:i+1])
    hnk_inv=HNK_inv2(hn_inv,data_pos_x[0:i+1],data_pos_x[i+1:i+2])
    hn_inv=hnk_inv
    print("n=",(i+1),",","alpha",(i+1),"=")
    print(alpha_n)

#the direct method     
def Hn(data): #define a function of computing Hn directly
    d=len(data) # the number of observation
    Km=np.zeros((d,d)) # K matrix named as Km
    I=np.eye(d) # d dimensions identity matrix
    for i in range(d):
        for j in range(d):
            Km[i][j]=k(data[i,:],data[j,:],sigma)
    hn=Km+(1/(2*C))*I # compute the matrix Hn-1   
    return (hn)

n=5  # assume we have 5 observations
for i in range(n):  # compute Hn by the direct method
    hn_inv=inv(Hn(data_pos_x[0:i+1]))
    print("n=",(i+1),",","inverse","of","H",(i+1),"=")
    print(hn_inv)      

for i in range(n): # compute alpha_n by the direct method
    hnk_inv=inv(Hn(data_pos_x[0:i+1]))
    alpha_n=alpha1(hnk_inv,data_pos_x[0:i+1])
    print("n=",(i+1),",","alpha",(i+1),"=") 
    print(alpha_n)

#problem 2 question c: check the performance of K=1 and K=15

#redefine R2
def R2(alpha_n,train,ap3): # R2 consists of three average parts:ap1,ap2,ap3 
    p1=np.zeros((len(train),1))
    p2=np.zeros((len(train),len(train)))
    
    for i in range(len(train)):
        p1[i,0]=k(train[i,:],train[i,:],sigma)  
    ap1=sum(p1)/len(train)
    
    for i in range(len(train)):
        for j in range(len(train)):
            p2[i,j]=k(train[i,:],train[j,:],sigma) 
    ap2=np.array(sum(np.matrix(p2)*np.matrix(alpha_n)))/ len(train)      
    
    r2=ap1-2*ap2+ap3
    return(r2)

#redefine DZ
def DZ(alpha_n,train,test,ap3): # assume dz consists of three parts:ap1,ap2,ap3
    ap1=k(test,test,sigma)
    p2=np.zeros((len(train),1))
    for i in range(len(train)):
        p2[i,0]=k(test,train[i,:],sigma)
    ap2=np.array(np.matrix(alpha_n.T)*np.matrix(p2))     
    dz=ap1-2*ap2+ap3
    return(dz) 
    
# split datasets into training and test
indice1=list(range(len(data_pos_x)))
indice2=list(range(len(data_neg_x)))
random.seed(6)
random.shuffle(indice1)
random.shuffle(indice2)
train= data_pos_x[indice1[:571],:]  # 90% for training
test= data_neg_x[indice2[:31],:]   # 10% for test

h1=np.matrix(k(train[0],train[0],sigma)+1/(2*C))
h1_inv=np.array(inv(h1)) # the inverse of H1
hn_inv=h1_inv    # the inverse of Hn starts from the inverse of H1
kk=1
tn2=int((len(train)-1)/kk)  # the number of training set
C=5  # the penalty factor
sigma=1 # the parameter of Gaussian Kernel   

#training starts at K=1
start_time6=time.time() 
for i in range(tn2): 
    hnk_inv=HNK_inv2(hn_inv,train[0:len(hn_inv)],train[len(hn_inv):len(hn_inv)+kk])
    hn_inv=hnk_inv
    print('the number of iteration=',i+1)
alpha_n=alpha1(hn_inv,train) # iterative alpha_n

p3=np.zeros((len(train),len(train))) # the third term in dz/R2 is a constant
for i in range(len(train)):
    for j in range(len(train)):
        p3[i,j]=k(train[i,:],train[j,:],sigma) 
ap3=sum(sum(np.outer(alpha_n,alpha_n.T)*p3)) # the third term in dz/R2 is a constant

r2=(R2(alpha_n,train,ap3))           # iterative R2 (n)
print("R2=",r2[0,0])  # output the trained radius R2  
print("the sum of alpha",sum(alpha_n)[0,0]) # output the trained alpha_n
print("K=1, Training time (s):",time.time()-start_time6) 
            
#compute the training accuracy when K=1
ta=0
for i in range(len(train)): 
    dz=DZ(alpha_n,train,train[i,:],ap3)
    if dz[0,0]<=r2[0,0]:
        ta+=1
        print("n=",i,"It's a target")
    else:
        print("n=",i,"It's an outlier")
train_accuracy=ta/len(train)
print("K=1, Training accuracy(%):",train_accuracy*100)

# compute the test accuracy when K=1
tt=0
for i in range(len(test)): 
    dz=DZ(alpha_n,train,test[i,:],ap3)
    if dz[0,0]<=r2[0,0]:
        tt+=1
        print("n=",i+1,"It's a target")
    else:
        print("n=",i+1,"It's an outlier")
test_accuracy=(len(test)-tt)/len(test)
print("K=1, Testing accuracy(%):",test_accuracy*100)

# if K=15

hn_inv=h1_inv    # the inverse of Hn starts from the inverse of H1
kk=15
tn2=int((len(train)-1)/kk)  # the number of training set
C=5  # the penalty factor
sigma=1 # the parameter of Gaussian Kernel   

#training starts at K=15
start_time7=time.time() 
for i in range(tn2): 
    hnk_inv=HNK_inv2(hn_inv,train[0:len(hn_inv)],train[len(hn_inv):len(hn_inv)+int(kk*tn2)])
    hn_inv=hnk_inv
    print('the number of iteration=',i+1)
alpha_n=alpha1(hn_inv,train) # iterative alpha_n

p3=np.zeros((len(train),len(train))) # the third term in dz/R2 is a constant
for i in range(len(train)):
    for j in range(len(train)):
        p3[i,j]=k(train[i,:],train[j,:],sigma) 
ap3=sum(sum(np.outer(alpha_n,alpha_n.T)*p3)) # the third term in dz/R2 is a constant

r2=R2(alpha_n,train,ap3)           # iterative R2 (n)
print("R2=",r2[0,0])  # output the trained radius R2  
print("the sum of alpha",sum(alpha_n)[0,0]) # output the trained alpha_n
print("K=15, Training time (s):",time.time()-start_time7) 
            
#compute the training accuracy when K=15
ta=0
for i in range(len(train)): 
    dz=DZ(alpha_n,train,train[i,:],ap3)
    if dz[0,0]<=r2[0,0]:
        ta+=1
        print("n=",i,"It's a target")
    else:
        print("n=",i,"It's an outlier")
train_accuracy=ta/len(train)
print("K=15, Training accuracy(%):",train_accuracy*100)

# compute the test accuracy when K=15
tt=0
for i in range(len(test)): 
    dz=DZ(alpha_n,train,test[i,:],ap3)
    if dz[0,0]<=r2[0,0]:
        tt+=1
        print("n=",i+1,"It's a target")
    else:
        print("n=",i+1,"It's an outlier")
test_accuracy=(len(test)-tt)/len(test)
print("K=15, Testing accuracy(%):",test_accuracy*100)

