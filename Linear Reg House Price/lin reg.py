#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:12:46 2020

@author: ankitjha
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset =pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values



# #fill missing values

# from sklearn.impute import SimpleImputer
# imputer=SimpleImputer(missing_values=np.nan,strategy="mean",axis=0)
# imputer =imputer.fit(X[:,1:3])
# X[:,1:3]=imputer.transform(X[:,1:3])



# #label encoding

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array(ct.fit_transform(X), dtype=np.float)
# from sklearn.preprocessing import LabelEncoder
# y = LabelEncoder().fit_transform(y)



#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3, random_state = 0)



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

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()