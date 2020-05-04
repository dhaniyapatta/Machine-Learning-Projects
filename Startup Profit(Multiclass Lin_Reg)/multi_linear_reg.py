#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:15:31 2020

@author: ankitjha
"""


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset =pd.read_csv('50_Startups.csv')
X= dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values



# #fill missing values

# from sklearn.impute import SimpleImputer
# imputer=SimpleImputer(missing_values=np.nan,strategy="mean",axis=0)
# imputer =imputer.fit(X[:,1:3])
# X[:,1:3]=imputer.transform(X[:,1:3])



# #label encoding

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding dummy trap we remove one dummy variable
#The python librbary although does this job for us, but for learning sake
X=X[:,1:]

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 0)



# # Feature Scaling

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1,1))


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# #just for checking 
# p=X_train[:,2]
# plt.scatter(p, y_train, color = 'red')
# plt.plot(p, regressor.predict(X_train), color = 'blue')
# plt.show()



































