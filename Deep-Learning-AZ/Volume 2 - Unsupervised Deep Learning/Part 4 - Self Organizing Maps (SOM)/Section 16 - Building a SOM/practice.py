import matplotlib.pyplot as pt
import numpy as np
import pandas as pd




dataset_practice =pd.read_csv("Credit_Card_Applications.csv")

X=dataset_practice.iloc[:,:-1]
y=dataset_practice.iloc[:,-1]

from sklearn.preprocessing  import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
sc = MinMaxScaler(feature_range = (0, 1))
X=sc.fit_transform(X)

from minisom import MiniSom

som=MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X ,num_iteration=50)

#Visivualing the result
from pylab import pcolor, bone,show,colorbar,plot
bone()
pcolor(som.distance_map().T)
colorbar()

markers=['o','s']
colors=['r','g']

for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',         
         markersize=10,
         markeredgewidth=2
        )
    
show()
    
   
mappings=som.win_map(X)
#frauds=np.concatenate((mappings[(3,1)],mappings[(4,10)]),axis=0)
frauds=mappings[(8,3)]

frauds=sc.inverse_transform(frauds)
