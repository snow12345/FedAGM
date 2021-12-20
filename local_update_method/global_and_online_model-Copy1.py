#!/usr/bin/env python
# coding: utf-8

# In[46]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import wandb
import sys

def set_epsilon(x,epsilon=2e-45):
    return epsilon+(1-epsilon)*x
def KD(input_p,input_q,T=1):
    p=F.softmax((input_p/T),dim=1)
    q=F.softmax((input_q/T),dim=1)
    result=((p*((p/q).log())).sum())/len(input_p)
    
    if not torch.isfinite(result):
        print('==================================================================')
        print('input_p')
        print(input_p)
        
        print('==================================================================')
        print('input_q')
        print(input_q)
        print('==================================================================')
        print('p')
        print(p)
        
        print('==================================================================')
        print('q')
        print(q)
        
        
        print('******************************************************************')
        print('p/q')
        print(p/q)
        
        print('******************************************************************')
        print('(p/q).log()')
        print((p/q).log())        
        
        print('******************************************************************')
        print('(p*((p/q).log())).sum()')
        print((p*((p/q).log())).sum())            
    
    return result
def JSD(input_p,input_q,T=1):
    p=F.softmax((input_p/T),dim=1)
    q=F.softmax((input_q/T),dim=1)
    
    p=set_epsilon(p)
    q=set_epsilon(q)
    
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    M=0.5*(p+q)   
    result=0.5*(criterion(M.log(),p)+criterion(M.log(),q))
    
    return result
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
    return loss


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
            global_outputs=self.global_save_output.get_outputs()

            
            
            
            
            
            intermediate_activation_loss=pod(    list_attentions_a=online_outputs[:-1],
                    list_attentions_b=global_outputs[:-1],
                    collapse_channels=self.args.collapse_channels,
                    normalize=self.args.pod_normalize,)/len(online_outputs)
            
            
            last_activation_loss=pod(    list_attentions_a=online_outputs[-1:],
                    list_attentions_b=global_outputs[-1:],
                    collapse_channels=self.args.collapse_channels,
                    normalize=self.args.pod_normalize,)/len(online_outputs)
            
            #criterion = torch.nn.KLDivLoss(reduction='batchmean')
            #logit_loss=(x.log(),x1)
            #mse=nn.MSELoss()
            
            #logit_loss=mse(x,x1)
            
            logit_loss=JSD(x1,x)
            #KD(input_p=x,input_q=x1,T=self.args.knowledge_temperature)
            if not torch.isfinite(intermediate_activation_loss):
                print('WARNING: non-finite intermediate_activation_loss, ending training ')
                exit(1)
            if not torch.isfinite(last_activation_loss):
                print('WARNING: non-finite last_activation_loss, ending training ')
                exit(1)
            if not torch.isfinite(logit_loss):
            #
                print('WARNING: non-finite logit_loss, ending training ')
                online_dict=self.online_model.state_dict()
                global_dict=self.global_model.state_dict()
                parameter_nan=False
                for n in (online_dict):
                    online_nan=torch.isnan(online_dict[n]).sum()
                    global_nan=torch.isnan(global_dict[n]).sum()
                    
                    if (online_nan>=1) or (global_nan>=1):
                        print("Nan parameter")
                        print("Name: "+n)
                        print("online")
                        print(online_nan>=1)
                        print(online_dict[n])
                        print("global")
                        print(global_nan>=1)
                        print(global_dict[n])
                        parameter_nan=True
               
                print("Exist nan parameter? :", parameter_nan)
                exit(1)
                
            wandb.log({'intermediate_activation_loss':intermediate_activation_loss})
            wandb.log({'last_activation_loss':last_activation_loss})
            wandb.log({'logit_loss':logit_loss})
                  
            activation_loss=self.args.lambda1*intermediate_activation_loss + self.args.lambda2*last_activation_loss + self.args.lambda3*logit_loss

            
            return x,activation_loss
            
        
        

            


# In[ ]:




