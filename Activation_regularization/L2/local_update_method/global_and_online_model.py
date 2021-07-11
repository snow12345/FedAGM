#!/usr/bin/env python
# coding: utf-8

# In[46]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy




def pod(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        a = torch.pow(a, 2)
        b = torch.pow(b, 2)
        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "pixel":
            pass        
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss
    return loss / len(list_attentions_a)


# In[17]:


class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
    
    def get_outputs(self):
        return self.outputs
    


def freeze_parameters(model):
    for n,m in model.named_parameters():
        m.grad=None
        m.requires_grad=False

class dual_model(nn.Module):
    def __init__(self,args,online_model,global_model):
        super(dual_model,self).__init__()
        self.args=args
        self.online_model=online_model
        self.global_model=copy.deepcopy(global_model)
        freeze_parameters(self.global_model)

        
        self.online_save_output = SaveOutput()
        self.global_save_output = SaveOutput()
    

        #현재 구현 상태: 각 Conv layer의 ouptut을 가지고 와서 그들의 distillation loss를 pod function을 이용해 구함
        #추후에 Resnet stage 단위로 바꿀 수 있다(Podnet paper:https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650086.pdf)
        self.online_hook_handles=[]
        for layer in self.online_model.modules():
            layer_name=layer._get_name()
            if self.args.regularization_unit in layer_name:
                handle = layer.register_forward_hook(self.online_save_output)
                self.online_hook_handles.append(handle)
        
        self.global_hook_handles=[]
        for layer in self.global_model.modules():
            layer_name=layer._get_name()
            if self.args.regularization_unit in layer_name:
                handle = layer.register_forward_hook(self.global_save_output)
                self.global_hook_handles.append(handle)
    

    
    
    def forward(self, x,online_target=False):
        if online_target==False:
            return self.online_model(x)
        
        else:
            x1=copy.deepcopy(x)   
            self.online_save_output.clear()
            self.global_save_output.clear()
         
            x=self.online_model(x)
            x1=self.global_model(x1)
            online_outputs=self.online_save_output.get_outputs()
            '''print('==============================')
            print('online outputs')
            print(online_outputs[0].requires_grad)'''
            global_outputs=self.global_save_output.get_outputs()
            '''print('==============================')
            print('global outputs')
            print(global_outputs[0].requires_grad)'''
            activation_loss=pod(    list_attentions_a=online_outputs,
                    list_attentions_b=global_outputs,
                    collapse_channels=self.args.collapse_channels,
                    normalize=self.args.pod_normalize,)
            
            return x,activation_loss
            
        
        

            


# In[ ]:




