import torch
import torch.nn as nn


__all__ = ['IL','CE']

class IL():
    def __init__(self,device,mean=True,gap=0.5,abs_thres=True):
        self.device=device
        self.mean=mean
        self.gap=gap
        self.abs_thres=abs_thres
    def __call__(self,outputs,labels):
        l=len(labels)
        sigmoid=torch.sigmoid(outputs)
        onehot=(torch.eye(10)[labels]).to(self.device)
        p_of_answer=(sigmoid*onehot).sum(axis=1)

        extend_p_of_answer=(self.gap*p_of_answer).unsqueeze(dim=1).expand(outputs.shape)
        if self.abs_thres==True:
            bigger_than_answer=(sigmoid>(self.gap))*(1-onehot)
        else:
            bigger_than_answer=(sigmoid>(extend_p_of_answer-self.gap))*(1-onehot)
        s=(((1-p_of_answer)**2))+(((sigmoid*bigger_than_answer)**2).sum(dim=1))/(bigger_than_answer.sum(dim=1)+1e-10)
        
        s=s.sum()
        if self.mean==True:
            s/=l
        return s



CE=nn.CrossEntropyLoss()