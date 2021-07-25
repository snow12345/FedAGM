#!/usr/bin/env python
# coding: utf-8

# In[2]:

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import numpy as np
from torch import nn
import copy
import torch
from models.mlp_head import MLPHead

def GlobalUpdate(args,device,trainset,testloader,LocalUpdate):
    model = get_model(args)
    model.to(device)
    wandb.watch(model)
    model.train()

    ## Get Predictor
    predictor = MLPHead(model.linear.in_features, 512, 512)
    predictor.to(device)
    wandb.watch(predictor)
    predictor.train()

    criterion = nn.CrossEntropyLoss().to(device)
    epoch_loss = []
    weight_saved = model.state_dict()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha

    for epoch in range(args.global_epochs):
        local_weight = []
        local_loss = []
        local_weight_predictor = []
        m = max(int(args.participation_rate * args.num_of_clients), 1)
        selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        print(f"This is global {epoch} epoch")
        for user in selected_user:
            local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                        batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
            weight, weight_predictor, loss = local_setting.train(net=copy.deepcopy(model).to(device), predictor=copy.deepcopy(predictor).to(device), epoch=epoch)
            local_weight.append(copy.deepcopy(weight))
            local_weight_predictor.append(copy.deepcopy(weight_predictor))
            local_loss.append(copy.deepcopy(loss))
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(1, len(local_weight)):
                FedAvg_weight[key] += local_weight[i][key]
            FedAvg_weight[key] /= len(local_weight)
        ## Update Predictor
        FedAvg_weight_predictor = copy.deepcopy(local_weight_predictor[0])
        for key in FedAvg_weight_predictor.keys():
            for i in range(1, len(local_weight_predictor)):
                FedAvg_weight_predictor[key] += local_weight_predictor[i][key]
            FedAvg_weight_predictor[key] /= len(local_weight_predictor)
        model.load_state_dict(FedAvg_weight)
        loss_avg = sum(local_loss) / len(local_loss)
        print(' Average loss {:.3f}'.format(loss_avg))
        loss_train.append(loss_avg)
        if epoch % args.print_freq == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    _, outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / float(total)))
            acc_train.append(100 * correct / float(total))

        model.train()

        wandb.log({args.mode + '_loss': loss_avg, args.mode + "_acc": acc_train[-1],'lr':this_lr})

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




