import torch


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


x = torch.randn(5, 5)  # requires_grad=False by default

print(x)

 y = torch.randn(5, 5)  # requires_grad=False by default
 
 print(y)
 
 z = torch.randn((5, 5), requires_grad=True)
 print(z)
  a = x + y
  print(a)
  print(a.requires_grad)
  
  b = a + z
  print(b.requires_grad)
  
  print(b)
  
'''  
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

'''