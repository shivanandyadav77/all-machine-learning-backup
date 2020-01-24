import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.parallel 
import torch.optim as optim
import torch.utils.data
from torch.autograd  import variable

# Importing the dataset
movies=pd.read_csv('ml-1m/movies.dat',sep='::' ,header=None,engine='python',encoding='latin-1')
user=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')



# Preparing the training set and the test set
traing_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
traing_set=np.array(traing_set ,dtype='int')

test_set=pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set=np.array(test_set,dtype='int')

# Getting the number of users and movies
nb_users=max(max(traing_set[:,0]),max(test_set[:,0]))
nb_movies=max(max(traing_set[:,1]),max(test_set[:,1]))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    
    new_data=[]
    for user_id in range(0,nb_users+1):
        movies_id=data[:,1][data[:,1]==user_id]
        ratings_id=data[:,2] [data[:,1]==user_id]
        new_rating=np.zeros(nb_movies)
        new_rating[movies_id-1]=ratings_id
        new_data.append(list(new_rating))
    return new_data
     
traing_set=convert(traing_set)
test_set=convert(test_set)
        
    




