#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:32:39 2018

@author: akshaychopra
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize



def gradient_descent(X, theta, y, alpha, max_iterations):
    
    final_theta=np.zeros(X.shape[1])
    final_theta=final_theta.reshape(X.shape[1],1)
    
    iterations=[]
    c=[]
    
    #Iterating max_iteration times to find the best theta    
    for i in range(max_iterations):
        
        z=np.dot(X,final_theta)
        hx= sigmoid(z)
        dz= hx-y
        grad= np.dot(np.transpose(X),dz)/X.shape[0]
        #grad= (1/m).*(X'*(hx-y)        
        
        final_theta= np.subtract(final_theta, alpha * grad)
        iterations.append(i)
        k=cost_function(X,final_theta,y)
        c.append(k)
        

        
    return final_theta
        


def cost_function(X,theta,y):
    
    #Calculating h(x)
    z=np.dot(X,theta)
    hx=sigmoid(z)
    cost= np.dot(np.transpose(-y),np.log(hx)) - np.dot((np.transpose(1 - y)), np.log(1 - hx))
    return cost/X.shape[0]
    
    
    
    
def sigmoid(X):
    return 1/(1 + np.exp(- X))

df=pd.read_csv("../Data/data.csv")

#Dropping the 'id' column from the dataset
df=df.drop("id",1)
df=df.drop("Unnamed: 32",1)

#Mapping M to 1 and B to 0 in the output Label DataFrame
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

#Split Data into training and test (70% and 30%)
train, test = train_test_split(df, test_size = 0.3)

#Training Data
train_x=train.loc[:,['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']]
train_y=train.loc[:,['diagnosis']]

#Testing Data
test_x=test.loc[:,['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']]
test_y=test.loc[:,['diagnosis']]

#Converting Traing and Test Data to numpy array
train_x=np.asarray(train_x)
train_y=np.asarray(train_y)
test_x=np.asarray(test_x)
test_y=np.asarray(test_y)

#Appending column of 1 to train_x and test_x for bias term
train_x = np.hstack([np.ones([train_x.shape[0],1]), train_x])
test_x = np.hstack([np.ones([test_x.shape[0],1]), test_x])


train_x=normalize(train_x, axis=0, norm='l1')
test_x=normalize(test_x, axis=0, norm='l1')


#Parameter vector theta
initial_theta=np.zeros(train_x.shape[1])

initial_theta=initial_theta.reshape(train_x.shape[1],1)

#Finding the initial error/cost 
cost = cost_function(train_x,initial_theta,train_y)


#Now, we will implement Gradient Descent to find the Optimal Parameters for the model
final_theta = gradient_descent(train_x, initial_theta, train_y, alpha = 1000.9, max_iterations = 50000)


cost1 = cost_function(test_x,final_theta,test_y)

print('hello')




