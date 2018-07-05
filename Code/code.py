#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:32:39 2018

@author: akshaychopra
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


Theta=np.zeros(train_x.shape[1]+1)