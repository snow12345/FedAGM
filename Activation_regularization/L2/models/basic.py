#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
from args import args
import torch.nn.functional as F
import copy
# In[1]:

class FC2(nn.Module):
    def __init__(self):
        super(FC2, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        
        

        self.fc1_global = nn.Linear(28*28, 200)
        self.fc2_global = nn.Linear(200, 100)
        self.fc3_global = nn.Linear(100, 10)
        self.freeze_model_global_weights()
        
        self.mse=nn.MSELoss(reduction='sum')
    def freeze_model_global_weights(self):
        for n, m in self.named_modules():
            if hasattr(m, "weight") and m.weight is not None and 'global' in n:
                print(f"==> No gradient to {n}.weight")
                m.weight.requires_grad = False
                if m.weight.grad is not None:
                    print(f"==> Setting gradient of {n}.weight to None")
                    m.weight.grad = None

                if hasattr(m, "bias") and m.bias is not None and 'global' in n:
                    print(f"==> No gradient to {n}.bias")
                    m.bias.requires_grad = False

                    if m.bias.grad is not None:
                        print(f"==> Setting gradient of {n}.bias to None")
                        m.bias.grad = None
    def sync_online_and_global(self):
        state_dict=self.state_dict()
        for key in state_dict:
            if 'global' in key:
                x=(key.split("_global"))
                online=(x[0]+x[1])
                state_dict[key]=state_dict[online]
        self.load_state_dict(state_dict)
    def forward(self, x,online_target=False):
        if online_target==False:
            x = x.view(-1, 28*28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = (self.fc3(x))
            return x
        else:
            total_num=0
            x1=copy.deepcopy(x)
            
   
            
            x = x.view(-1, 28*28)
            x1 = x1.view(-1, 28*28)
            
            x = F.relu(self.fc1(x))
            x1 = F.relu(self.fc1_global(x1))
            diff=self.mse(x,x1)
            total_num+=x1.numel()
            
            x = F.relu(self.fc2(x))
            x1 = F.relu(self.fc2_global(x1))
            diff+=self.mse(x,x1)
            total_num+=x1.numel()
            
            
            x = (self.fc3(x))
            x1 = (self.fc3_global(x1))
            diff+=self.mse(x,x1)
            total_num+=x1.numel()
            
            
            
            diff/=total_num
            
            return x,diff

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5,padding=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 394)
        self.fc2 = nn.Linear(394, 192)
        self.fc3 = nn.Linear(192, 10)
        
        
        
        self.conv1_global = nn.Conv2d(3, 64, 5,padding=1)
        self.pool_global = nn.MaxPool2d(2, 2)
        self.conv2_global = nn.Conv2d(64, 64, 5,padding=1)
        self.fc1_global = nn.Linear(64 * 6 * 6, 394)
        self.fc2_global = nn.Linear(394, 192)
        self.fc3_global = nn.Linear(192, 10)
        self.freeze_model_global_weights()
        
        self.mse=nn.MSELoss(reduction='sum')
    def freeze_model_global_weights(self):
        for n, m in self.named_modules():
            if hasattr(m, "weight") and m.weight is not None and 'global' in n:
                print(f"==> No gradient to {n}.weight")
                m.weight.requires_grad = False
                if m.weight.grad is not None:
                    print(f"==> Setting gradient of {n}.weight to None")
                    m.weight.grad = None

                if hasattr(m, "bias") and m.bias is not None and 'global' in n:
                    print(f"==> No gradient to {n}.bias")
                    m.bias.requires_grad = False

                    if m.bias.grad is not None:
                        print(f"==> Setting gradient of {n}.bias to None")
                        m.bias.grad = None
    def sync_online_and_global(self):
        state_dict=self.state_dict()
        for key in state_dict:
            if 'global' in key:
                x=(key.split("_global"))
                online=(x[0]+x[1])
                state_dict[key]=state_dict[online]
        self.load_state_dict(state_dict)
    def forward(self, x,online_target=False):
        if online_target==False:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*6*6)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = (self.fc3(x))
            return x
        else:
            total_num=0
            x1=copy.deepcopy(x)
            
            
            x = self.pool(F.relu(self.conv1(x)))
            x1 = self.pool_global(F.relu(self.conv1_global(x1)))
            diff=self.mse(x,x1)
            total_num+=x1.numel()
            
            x = self.pool(F.relu(self.conv2(x)))
            x1 = self.pool_global(F.relu(self.conv2_global(x1)))
            diff+=self.mse(x,x1)
            total_num+=x1.numel()
            
            x = x.view(-1, 64*6*6)
            x1 = x1.view(-1, 64*6*6)
            
            x = F.relu(self.fc1(x))
            x1 = F.relu(self.fc1_global(x1))
            diff+=self.mse(x,x1)
            total_num+=x1.numel()
            
            x = F.relu(self.fc2(x))
            x1 = F.relu(self.fc2_global(x1))
            diff+=self.mse(x,x1)
            total_num+=x1.numel()
            
            
            x = (self.fc3(x))
            x1 = (self.fc3_global(x1))
            diff+=self.mse(x,x1)
            total_num+=x1.numel()
            
            
            
            diff/=total_num
            
            return x,diff


# In[ ]:




