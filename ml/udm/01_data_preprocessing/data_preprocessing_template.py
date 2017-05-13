#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:45:21 2017

@author: anh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Data.csv')

# : take all the lines, :-1 all the collumn except the last one (Purchase)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
                
# take care of all missing data, replace missing data with mean value
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

# categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
# use dummy variables make contries to 3 other columns (dummy ecoding)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# spliting data into training set and test set
# from sklearn.cross_validation import train_test_split 
# cross_validation: This module will be removed in 0.20. use model_selection instead
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# features scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

