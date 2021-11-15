#!/usr/bin/env python
# coding: utf-8

# In[2]:

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import numpy as np
from torch import nn
import copy
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import DatasetSplit
from global_update_method.distcheck import check_data_distribution
import umap.umap_ as umap
from mpl_toolkits import mplot3d
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from torch.utils.data import DataLoader
from utils import log_ConfusionMatrix_Umap, log_acc
from utils import *
from utils import CenterUpdate

def GlobalUpdate(args,device,trainset,testloader,LocalUpdate):
    model = get_model(args)
    model.to(device)
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    epoch_loss = []
    weight_saved = model.state_dict()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha

    
    
    delta_t=copy.deepcopy(model.state_dict())
    v_t=copy.deepcopy(delta_t)
    for key in delta_t.keys():
        delta_t[key]*=0
        v_t[key]=delta_t[key]+(args.tau**2)
    this_server_lr=args.g_lr
    ideal_model=copy.deepcopy(model)
    ideal_delta_t=copy.deepcopy(delta_t)
    ideal_v_t=copy.deepcopy(v_t)
    
    
    
    
    for epoch in range(args.global_epochs):
        global_weight = copy.deepcopy(model.state_dict())
        wandb_dict={}
        num_of_data_clients=[]
        local_weight = []
        local_loss = []
        local_delta = []
        m = max(int(args.participation_rate * args.num_of_clients), 1)
        selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        print(f"This is global {epoch} epoch")
        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                        batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
            local_delta.append(delta)
            client_ldr_train = DataLoader(DatasetSplit(trainset, dataset[user]), batch_size=args.batch_size, shuffle=True)
        total_num_of_data_clients=sum(num_of_data_clients)            
            
        client_weight = copy.deepcopy(local_weight[0])
        delta_client_mean = copy.deepcopy(model.state_dict())
        x_t = copy.deepcopy(model.state_dict())
        for key in client_weight.keys():
            for i in range(len(local_weight)):
                if i==0:
                    client_weight[key]*=num_of_data_clients[i]
                else:
                    client_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            client_weight[key] /= total_num_of_data_clients
            delta_client_mean[key]=client_weight[key]-x_t[key]
            delta_t[key]=delta_t[key]*args.beta_1 + delta_client_mean[key]*(1-args.beta_1)
            v_t[key]=v_t[key]*args.beta_2+(delta_t[key]*delta_t[key])*(1-args.beta_2)
            x_t[key]+=this_server_lr*delta_t[key]/((v_t[key]**0.5)+args.tau)
                 
            
        prev_model_weight = copy.deepcopy(model.state_dict())
        current_model_weight = copy.deepcopy(x_t)
        model.load_state_dict(x_t)
        
        
        if args.compare_with_center>0:
            if args.compare_with_center ==1:
                idxs=None
            elif args.compare_with_center ==2:
                idxs=[]
                for user in selected_user:
                    idxs+=dataset[user]
            ideal_x_t = copy.deepcopy(ideal_model.state_dict())
            centerupdate = CenterUpdate(args=args,lr = this_lr,iteration_num = len(client_ldr_train)*args.local_epochs,device =device,batch_size=args.batch_size*m,dataset =trainset,idxs=idxs,num_of_participation_clients=m)
            center_weight = centerupdate.train(net=copy.deepcopy(model).to(device))  
            ideal_weight = centerupdate.train(net=copy.deepcopy(ideal_model).to(device))  
            for key in ideal_weight.keys():
                ideal_delta_t[key] = ideal_delta_t[key]*args.beta_1 + (ideal_weight[key]-ideal_x_t[key])*(1-args.beta_1)
                ideal_v_t[key] = ideal_v_t[key]*args.beta_2+(ideal_delta_t[key]*ideal_delta_t[key])*(1-args.beta_2)
                ideal_x_t[key] +=this_server_lr*ideal_delta_t[key]/((ideal_v_t[key]**0.5)+args.tau)
                
            ideal_model.load_state_dict(ideal_x_t)
            divergence_from_central_update = calculate_divergence_from_center(args, center_weight, client_weight)
            divergence_from_central_model = calculate_divergence_from_center(args, ideal_x_t, x_t)
            wandb_dict[args.mode + "_divergence_from_central_update"] = divergence_from_central_update  
            wandb_dict[args.mode + "_divergence_from_central_model"] = divergence_from_central_model

        if args.analysis:
            ## calculate delta cv
            #delta_cv = calculate_delta_cv(args, copy.deepcopy(model), copy.deepcopy(local_delta), num_of_data_clients)

            ## calculate delta variance
            #delta_variance = calculate_delta_variance(args, copy.deepcopy(local_delta), num_of_data_clients)

            ## Calculate distance from Centralized Optimal Point
            #checkpoint_path = '/data2/geeho/fed/{}/{}/best.pth'.format(args.set, 'centralized')
            #divergence_from_centralized_optimal = calculate_divergence_from_optimal(args, checkpoint_path,
                                                                                   # x_t)
            checkpoint_path = './data/saved_model/fed/CIFAR10/centralized/Fedavg/_best.pth'
            cosinesimilarity=calculate_cosinesimilarity_from_optimal(args, checkpoint_path, current_model_weight, prev_model_weight)
            wandb_dict[args.mode + "_cosinesimilarity"] = cosinesimilarity
            ## Calculate Weight Divergence
            #wandb_dict[args.mode + "_delta_cv"] = delta_cv
            #wandb_dict[args.mode + "_delta_gnsr"] = 1 / delta_cv
            #wandb_dict[args.mode + "_delta_variance"] = delta_variance
            #wandb_dict[args.mode + "_divergence_from_centralized_optimal"] = divergence_from_centralized_optimal
            
        
        
        
        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ',num_of_data_clients)
        print(' Average loss {:.3f}'.format( loss_avg))
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
        wandb_dict[args.mode + "_acc"]=acc_train[-1]
        wandb_dict[args.mode + '_loss']= loss_avg
        wandb_dict['lr']=this_lr
        wandb.log(wandb_dict)

        this_lr *= args.learning_rate_decay
        #this_lr=args.lr/((epoch+1)**0.5)
        
        
        
        
        if args.alpha_mul_epoch == True:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch == True:
            this_alpha = args.alpha / (epoch + 1)


    print('loss_train')
    print(loss_train)

    print('acc_train')
    print(acc_train)


# In[ ]:




