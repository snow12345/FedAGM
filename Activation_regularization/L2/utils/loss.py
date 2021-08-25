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

        extend_p_of_answer=(p_of_answer*self.gap).unsqueeze(dim=1).expand(outputs.shape)
        if self.abs_thres==True:
            bigger_than_answer=(sigmoid>(self.gap))*(1-onehot)
            print("sigmoid",sigmoid[0])
            print("bigger_than_answer",bigger_than_answer[0])
        else:
            bigger_than_answer=(sigmoid>(extend_p_of_answer))*(1-onehot)
        pos=(((1-p_of_answer)**2))
        neg=(((sigmoid*bigger_than_answer)**2).sum(dim=1))/9##((bigger_than_answer.sum(dim=1)+1e-10))
        s=pos+neg
        
        s=s.sum()
        if self.mean==True:
            s/=l
        return s



CE=nn.CrossEntropyLoss()