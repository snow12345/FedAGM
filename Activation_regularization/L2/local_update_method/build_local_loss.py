
import torch
import torch.nn as nn
def individual_loss(outputs,labels,mean=True,gap=0.5):
    device=labels.device
    l=len(labels)
    sigmoid=torch.sigmoid(outputs)
    onehot=(torch.eye(10)[labels]).to(device)
    p_of_answer=(sigmoid*onehot).sum(axis=1)
    
    extend_p_of_answer=p_of_answer.unsqueeze(dim=1).expand(outputs.shape)
    bigger_than_answer=sigmoid*(sigmoid>(extend_p_of_answer-gap))
    s=(((1-p_of_answer)**2)).sum()+((bigger_than_answer**2).sum()-(p_of_answer**2).sum())/9
    if mean==True:
        s/=l
    return s










def build_loss(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.loss == 'CE':
        return nn.CrossEntropyLoss()
    elif args.loss=='Individual_loss':
        return individual_loss
    else:
        assert False
