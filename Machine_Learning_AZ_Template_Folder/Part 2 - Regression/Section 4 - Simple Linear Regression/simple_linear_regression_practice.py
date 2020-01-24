# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:48:38 2019

@author: shivayad
"""

import numpy as np
import  pandas as pd
import  matplotlib.pyplot as pt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

print(X)
print("---------------------")
print(y)

print("----------------")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

linearregression=LinearRegression()
linearregression.fit(X_train,y_train)


y_pred=linearregression.predict(X_test)

pt.scatter(X_train,y_train,color="red")
pt.plot(X_train,linearregression.predict(X_train),color="blue")
pt.xlabel("year of experience")
pt.ylabel("salary")
pt.title("Salary vs Experience (Training set)")
pt.show()




pt.scatter(X_test,y_test,color="red")
pt.plot(X_test,linearregression.predict(X_test),color="blue")
pt.xlabel("year of experience")
pt.ylabel("salary")
pt.title("Salary vs Experience (Testing set)")
pt.show()