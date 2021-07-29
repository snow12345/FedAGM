import torch
import torch.nn as nn


__all__ = ['IL','CE']

class IL():
    def __init__(self,device,mean=True,gap=0.5):
        self.device=device
        self.mean=mean
        self.gap=gap
    def __call__(self,outputs,labels):
        l=len(labels)
        sigmoid=torch.sigmoid(outputs)
        onehot=(torch.eye(10)[labels]).to(self.device)
        p_of_answer=(sigmoid*onehot).sum(axis=1)

        extend_p_of_answer=(self.gap*p_of_answer).unsqueeze(dim=1).expand(outputs.shape)

        bigger_than_answer=sigmoid*(sigmoid>(extend_p_of_answer))

        s=(((1-p_of_answer)**2)).sum()+((bigger_than_answer**2).sum()-(p_of_answer**2).sum())/9

        if self.mean==True:
            s/=l
        return s



CE=nn.CrossEntropyLoss()