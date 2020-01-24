# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:55:32 2019

@author: shivayad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import Imputer,Normalizer,StandardScaler,OneHotEncoder,LabelEncoder
'''from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder'''


dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values


imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

labelencoder_X=LabelEncoder()

X[:,0]=labelencoder_X.fit_transform(X[:,0])

onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()


labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc_X=StandardScaler();

X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)








