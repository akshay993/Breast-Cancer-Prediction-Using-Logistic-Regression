#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



def init():
    
    df=pd.read_csv("../Data/data.csv")
    
    
    #Dropping the 'id' column from the dataset
    df=df.drop("id",1)
    df=df.drop("Unnamed: 32",1)
    
    #Mapping M to 1 and B to 0 in the output Label DataFrame
    df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
    
    
    #Split Data into training and test (70% and 30%)
    train, test = train_test_split(df, test_size = 0.3, random_state=1)
        
    
    #Training Data
    train_x=train.loc[:,'radius_mean' : 'fractal_dimension_worst']
    train_y=train.loc[:,['diagnosis']]
    
    #Testing Data
    test_x=test.loc[:,'radius_mean' : 'fractal_dimension_worst']
    test_y=test.loc[:,['diagnosis']]
    
    #Converting Traing and Test Data to numpy array
    train_x=np.asarray(train_x)
    train_y=np.asarray(train_y)
    test_x=np.asarray(test_x)
    test_y=np.asarray(test_y)
    
    #Calling the model function to train a Logistic Regression Model on Training Data
    d= model(train_x.T, train_y.T, num_of_iterations=10000, alpha=0.000001)
    
        
    costs=d["costs"]
    w=d["w"]
    b=d["b"]
    
    #Drawing the plot between cost and number of iterations
    plt.plot(costs)
    plt.title("Cost vs #Iterations")
    plt.xlabel("Number of Iterations ( * 10)")
    plt.ylabel("Cost")
    
    
    #Now, calculating the accuracy on Training and Test Data
    Y_prediction_train = predict(train_x.T, w, b)
    Y_prediction_test = predict(test_x.T, w, b)
    
    
    
    print("\nTrain accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y.T)) * 100))
    
    print("\nTest accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y.T)) * 100))
    
    
    
    

#Function to initialize the weights and bias
def initialize(m):
    
    w = np.zeros((m,1))
    b = 0
    
    return w , b
    
#Function to calculate sigmoid of x    
def sigmoid(X):
    return 1/(1 + np.exp(- X))    


#Function for doing forward and back propogation
def propogate(X, Y, w, b):
    
    m = X.shape[1] #Number of training examples

    #Forward Propogation, calculating the cost
    Z = np.dot(w.T, X) + b;    
    A = sigmoid(Z)
    cost= -(1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    
    #Back Propogation , calculating the gradients
    dw = (1/m)* np.dot(X, (A-Y).T)
    db = (1/m)* np.sum(A-Y)
    
    grads= {"dw" : dw, "db" : db}
    
    return grads, cost
    
    


#Function for performing Grdient Descent
def optimize(X, Y, w, b, num_of_iterations, alpha):
    
    costs=[] 
    
    for i in range(num_of_iterations):
 
        grads, cost = propogate(X, Y, w, b)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - alpha * dw
        b = b - alpha * db
        
        #Storing tthe cost at interval of every 10 iterations
        if i% 10 == 0:
            costs.append(cost)
            print("cost after %i iteration : %f" % (i, cost))
            
            
    parameters = {"w":w, "b":b}
    grads = {"dw":dw, "db":db}
    
    
    return parameters, grads, costs


#Function for doing the predictions on the data set (mapping probabilities to 0 or 1)
def predict(X, w, b):
    
    m = X.shape[1] #Number of training examples
    
    y_prediction =  np.zeros((1,m))
    
    w = w.reshape(X.shape[0], 1)
    
    A=sigmoid(np.dot(w.T, X)+b)
    
    
    for i in range(A.shape[1]):
        
        if(A[0,i]<0.5):
            y_prediction[0,i]=0
        else:
            y_prediction[0,i]=1
            
    
    return y_prediction


#Function for calculating the Logistic Regression Model
def model(Xtrain, Ytrain, num_of_iterations, alpha):
    
    dim = Xtrain.shape[0] #Number of features
    
    w,b = initialize(dim)
    
    parameters, grads, costs = optimize(Xtrain, Ytrain, w, b, num_of_iterations, alpha) 
    
    w = parameters["w"]
    b = parameters["b"]
        
    
    d={"w":w, "b":b, "costs": costs}
    
    return d
    


#Calling the init function to start the program
init()





