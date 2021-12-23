#!/usr/bin/env python
# coding: utf-8

# In[2]:

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import numpy as np
from torch import nn
import copy
import torch
from utils import log_ConfusionMatrix_Umap, log_acc, get_activation
from global_update_method.distcheck import check_data_distribution_aug

import matplotlib.pyplot as plt
from utils import DatasetSplit
from global_update_method.distcheck import check_data_distribution
from torch.utils.data import DataLoader
from utils import calculate_delta_cv,calculate_delta_variance, calculate_divergence_from_optimal,calculate_divergence_from_center
from utils import CenterUpdate
from utils import *

def GlobalUpdate(args, device, trainset, testloader, LocalUpdate):
    model = get_model(args)
    model.to(device)
    wandb.watch(model)
    model.train()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha

    global_delta = copy.deepcopy(model.state_dict())
    for key in global_delta.keys():
        global_delta[key] = torch.zeros_like(global_delta[key])

    global_h = copy.deepcopy(model.state_dict())
    for key in global_h.keys():
        global_h[key] = torch.zeros_like(global_h[key])

    local_g = copy.deepcopy(model.state_dict())
    for key in local_g.keys():
        local_g[key] = torch.zeros_like(local_g[key]).to('cpu')
    local_deltas = {net_i: copy.deepcopy(local_g) for net_i in range(args.num_of_clients)}

    m = max(int(args.participation_rate * args.num_of_clients), 1)

    for epoch in range(args.global_epochs):
        wandb_dict = {}
        num_of_data_clients = []
        local_K = []

        local_weight = []
        local_loss = []
        local_delta = []
        # print('global delta of linear weight', global_delta['linear.weight'])
        global_weight = copy.deepcopy(model.state_dict())
        if (epoch == 0) or (args.participation_rate < 1):
            selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        else:
            pass
        print(f"This is global {epoch} epoch")

        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                        batch_size=args.batch_size, dataset=trainset, idxs=dataset[user],
                                        alpha=this_alpha, local_deltas=local_deltas)
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device), user=user)
            local_K.append(local_setting.K)

            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
            # Store local delta
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
            local_delta.append(delta)

            client_ldr_train = DataLoader(DatasetSplit(trainset, dataset[user]), batch_size=args.batch_size,
                                          shuffle=True)

        total_num_of_data_clients = sum(num_of_data_clients)
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i == 0:
                    FedAvg_weight[key] *= num_of_data_clients[i]
                else:
                    FedAvg_weight[key] += local_weight[i][key] * num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients

        for key in global_delta.keys():
            for i in range(len(local_delta)):
                if i == 0:
                    global_delta[key] = local_delta[0][key]
                else:
                    global_delta[key] += local_delta[i][key]
            global_delta[key] = global_delta[key] * (-1 * args.alpha / args.num_of_clients)
            global_h[key] = global_h[key] + global_delta[key]
            global_weight[key] = FedAvg_weight[key] - global_h[key] / args.alpha

        # Global weight update
        model.load_state_dict(global_weight)
        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ', num_of_data_clients)
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
        wandb_dict[args.mode + "_acc"]=acc_train[-1]
        wandb_dict[args.mode + '_loss']= loss_avg
        wandb_dict['lr']=this_lr
        wandb.log(wandb_dict)
        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch == True:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch == True:
            this_alpha = args.alpha / (epoch + 1)