#!/usr/bin/env python
# coding: utf-8

# In[2]:

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import numpy as np
from torch import nn
import copy
import torch
from torch.optim import Optimizer#, required


def compare_del(input1,input2,device,section=10):
    count=torch.zeros(section).to(device)
    total=torch.zeros(section).to(device)
    result=torch.zeros(section).to(device)
    mag=torch.zeros(section).to(device)

    for i in range(len(input1)):
        flatten_input1=input1[i].view(-1).to(device)
        flatten_input2=input2[i].view(-1).to(device)
        input1_len=len(flatten_input1)
        section_len=int(input1_len/section)
        val,idx=torch.topk(abs(flatten_input1),input1_len)
        sign_input1=torch.sign(flatten_input1)
        sign_input2=torch.sign(flatten_input2)
        for j in range(section-1):
            total[j]+=section_len
            for k in idx[section_len*j:section_len*(j+1)]:
                if sign_input1[k]==sign_input2[k]:
                    count[j]+=1
                mag[j]+=(flatten_input1[k]-flatten_input2[k])**2
        last=idx[section_len*(section-1):]
        #print(last)
        total[-1]+=len(last)
        for k in last:
            if sign_input1[k]==sign_input2[k]:
                count[-1]+=1
            mag[-1]+=(flatten_input1[k]-flatten_input2[k])**2
    for a in range(len(count)):
        result[a]=count[a]/float(total[a])
        mag[a]/=float(total[a])
    return result,mag
def get_dict_value_list(x,mul=1):
    return ([x[key]*mul for key in x])
def get_dict_value_tensor(x):
    return torch.stack([x[key] for key in x])
class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=-1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None,update=True):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        gradient=[]
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                gradient.append(d_p)
                if update==True:
                    p.add_(d_p, alpha=-group['lr'])
                    

        return loss,(gradient)

def get_centralized_gradient(trainloader,net,device):

    optimizer=SGD(net.parameters(),lr=0.0)
    criterion=nn.CrossEntropyLoss()
    gradient=[]
    for data in trainloader:
        optimizer.zero_grad()
        images, labels = data[0].to(device), data[1].to(device)
        outputs=net(images)

        loss = criterion(outputs, labels)
        loss.backward()
        _,this_gradient=optimizer.step(update=False)
        gradient.append(this_gradient)
        
    mean_gradient=gradient[0]
    for i in range(len(mean_gradient)):
        for j in range(1,len(gradient)):

            mean_gradient[i]+=gradient[j][i]
        mean_gradient[i]/=len(trainloader)
    return mean_gradient

def GlobalUpdate(args,device,trainset,testloader,LocalUpdate):
    CIFAR_Centralized_trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                              shuffle=True, num_workers=8)
    model = get_model(args)
    model.to(device)
    wandb.watch(model)
    model.train()
    epoch_loss = []
    weight_saved = model.state_dict()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha
    global_delta = copy.deepcopy(model.state_dict())
    for key in global_delta.keys():
        global_delta[key] = torch.zeros_like(global_delta[key])

    for epoch in range(args.global_epochs):
        wandb_dict={}
        num_of_data_clients=[]
        local_K=[]
        
        local_weight = []
        local_loss = []
        local_delta = []
        #print('global delta of linear weight', global_delta['linear.weight'])
        global_weight = copy.deepcopy(model.state_dict())
        m = max(int(args.participation_rate * args.num_of_clients), 1)
        selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        print(f"This is global {epoch} epoch")
        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                        batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device), delta=global_delta)
            local_K.append(local_setting.K)
            #weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
            ## store local delta
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
                delta[key]*=num_of_data_clients[-1]/local_K[-1]
            local_delta.append(delta)
            
        total_num_of_data_clients=sum(num_of_data_clients)
        
        
        centralized_gradient=get_centralized_gradient(CIFAR_Centralized_trainloader,net=copy.deepcopy(model).to(device),device=device)
        
        sign_correct=[]
        distance_client_center=[]
        with torch.no_grad():
            for i,delta in enumerate(local_delta):
                this_local_delta=get_dict_value_list(delta,mul=1/(total_num_of_data_clients * args.local_epochs))
                this_gradient=copy.deepcopy(centralized_gradient)
                this_sign_correct,this_distance=compare_del(this_local_delta,this_gradient,device=device)
                sign_correct.append(this_sign_correct)
                distance_client_center.append(this_distance)
            
            
            sign_correct=(torch.stack(sign_correct)).mean(dim=0)
            distance_client_center=(torch.stack(distance_client_center)).mean(dim=0)
            for i in range(len(sign_correct)):
                wandb_dict[args.mode + '_sign_correct_'+str(i)]=sign_correct[i].cpu()
                wandb_dict[args.mode + "_distance_client_center_"+str(i)]=distance_client_center[i].cpu()

            
        
        
        
        
        
        
        
        
        
        
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i==0:
                    FedAvg_weight[key]*=num_of_data_clients[i]
                else:                       
                    FedAvg_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients
        global_delta = copy.deepcopy(local_delta[0])
        
        
        
        K_mean=sum(local_K)/len(local_K)
        for key in global_delta.keys():
            for i in range(len(local_delta)):
                global_delta[key] += local_delta[i][key]
            global_delta[key] = global_delta[key] / (-1 * total_num_of_data_clients * args.local_epochs * this_lr)
            #global_delta[key] = global_delta[key] / float((-1 * len(local_delta)))
            global_lr = args.g_lr
            #global_lr = args.g_lr
            #print('global_lr', global_lr)
            global_weight[key] = global_weight[key] - global_lr * global_delta[key]
            #print((FedAvg_weight[key] == global_weight[key]).all())
        ## global weight update
        model.load_state_dict(FedAvg_weight)
        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ',num_of_data_clients)  
        print(' Average loss {:.3f}'.format(loss_avg))
        loss_train.append(loss_avg)
        if epoch % args.print_freq == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / float(total)))
            acc_train.append(100 * correct / float(total))

        model.train()
        wandb_dict[args.mode + '_loss']= loss_avg
        wandb_dict[args.mode + "_acc"]=acc_train[-1]
        wandb_dict['lr']=this_lr
        wandb.log(wandb_dict)

        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch == True:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch == True:
            this_alpha = args.alpha / (epoch + 1)


    print('loss_train')
    print(loss_train)

    print('acc_train')
    print(acc_train)


# In[ ]:




