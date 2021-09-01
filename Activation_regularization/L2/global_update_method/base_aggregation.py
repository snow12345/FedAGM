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

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def log_acc(model,testloader,args,wandb_dict,name):
    model.eval()
    device=next(model.parameters()).device
    first=True
    with torch.no_grad():
        for data in testloader:
            activation = {}
            model.layer4.register_forward_hook(get_activation('layer4',activation))
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if first:
                features=activation['layer4'].view(len(images),-1)
                saved_labels=labels
                saved_pred=predicted
                first=False
            else:
                features=torch.cat((features,activation['layer4'].view(len(images),-1)))
                saved_labels=torch.cat((saved_labels,labels))
                saved_pred=torch.cat((saved_pred,predicted))


        

        saved_labels=saved_labels.cpu()
        saved_pred=saved_pred.cpu()

        f1=metrics.f1_score(saved_labels,saved_pred,average='weighted')
        acc=metrics.accuracy_score(saved_labels,saved_pred)
        wandb_dict[name+" f1"]=f1
        wandb_dict[name+" acc"]=acc

        
    model.train()
    return acc


def log_ConfusionMatrix_Umap(model,testloader,args,wandb_dict,name):
    model.eval()
    device=next(model.parameters()).device
    first=True
    with torch.no_grad():
        for data in testloader:
            activation = {}
            model.layer4.register_forward_hook(get_activation('layer4',activation))
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if first:
                features=activation['layer4'].view(len(images),-1)
                saved_labels=labels
                saved_pred=predicted
                first=False
            else:
                features=torch.cat((features,activation['layer4'].view(len(images),-1)))
                saved_labels=torch.cat((saved_labels,labels))
                saved_pred=torch.cat((saved_pred,predicted))


        

        saved_labels=saved_labels.cpu()
        saved_pred=saved_pred.cpu()

        #plt.figure()
        f1=metrics.f1_score(saved_labels,saved_pred,average='weighted')
        acc=metrics.accuracy_score(saved_labels,saved_pred)
        cm=metrics.confusion_matrix(saved_labels,saved_pred)
        wandb_dict[name+" f1"]=f1
        wandb_dict[name+" acc"]=acc
        plt.figure(figsize=(20,20))
        #wandb_dict[args.mode+name+" f1"]=f1
        #wandb_dict[args.mode+name+" acc"]=acc
        fig, ax=plot_confusion_matrix(cm,class_names=classes,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                figsize=(16,16)
                                )
        ax.margins(2,2)


        
        
        wandb_dict[name+" confusion_matrix"]=wandb.Image(fig)
        
        
        y_test = np.asarray(saved_labels.cpu())

        reducer=umap.UMAP(random_state=0,n_components=args.umap_dim)
        embedding=reducer.fit_transform(features.cpu())



        plt.figure(figsize=(20,20))
        

        if args.umap_dim==3:
            ax=plt.axes(projection=('3d'))
        else:
            ax=plt.axes()

        for i in range(len(classes)):
            y_i = (y_test == i)
            scatter_input=[embedding[y_i,k] for k in range(args.umap_dim)]
            ax.scatter(*scatter_input, label=classes[i])
        plt.legend(loc=4)
        plt.gca().invert_yaxis()

        wandb_dict[name+" umap"]=wandb.Image(plt)



    model.train()
    return acc

def get_activation(name,activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


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
    m = max(int(args.participation_rate * args.num_of_clients), 1)

    for epoch in range(args.global_epochs):
        wandb_dict={}
        num_of_data_clients=[]
        local_weight = []
        local_loss = []
        if (epoch==0) or (args.participation_rate<1) :
            selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        else:
            pass 
        print(f"This is global {epoch} epoch")
                        
                        
        if (args.umap==True) and (epoch%args.umap_freq==0):
            if epoch % args.print_freq == 0:                        
                global_acc=log_ConfusionMatrix_Umap(copy.deepcopy(model),testloader,args,wandb_dict,name="global model_before local training")                        
                        
                        
        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                        batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
            client_ldr_train = DataLoader(DatasetSplit(trainset, dataset[user]), batch_size=args.batch_size, shuffle=True)
                        
            
            if (args.umap==True) and (epoch%args.umap_freq==0):
                if epoch % args.print_freq == 0:
                    name="client"+str(user)
                    if (epoch==0) or (args.participation_rate<=1) :
                        
                        data_distribution=check_data_distribution(client_ldr_train)
                        plt.figure(figsize=(20,20))
                        plt.bar(range(len(data_distribution)),data_distribution)
                        wandb_dict[name+"data_distribution"]=wandb.Image(plt)                        
                    else:
                        pass 
                    wandb_dict[name+"local loss"]=loss
                    this_model=copy.deepcopy(model)
                    this_model.load_state_dict(weight)
                    log_acc(this_model,client_ldr_train,args,wandb_dict,name=name+" local")  
                    log_ConfusionMatrix_Umap(this_model,testloader,args,wandb_dict,name=name)                             
                        
                        
                        
        total_num_of_data_clients=sum(num_of_data_clients)
                                     
        
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i==0:
                    FedAvg_weight[key]*=num_of_data_clients[i]
                else:                       
                    FedAvg_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients
        model.load_state_dict(FedAvg_weight)
        loss_avg = sum(local_loss) / len(local_loss)
                                       
                                       
        print(' num_of_data_clients : ',num_of_data_clients)                                   
        print(' Average loss {:.3f}'.format(loss_avg))
        loss_train.append(loss_avg)
        if (args.t_sne==True) and (epoch%args.t_sne_freq==0):
            if epoch % args.print_freq == 0:
                model.eval()
                correct = 0
                total = 0
                first=True
                with torch.no_grad():
                    for data in testloader:
                        activation = {}
                        model.layer4.register_forward_hook(get_activation('layer4',activation))
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = model(images)
                        if first:
                            features=activation['layer4'].view(len(images),-1)
                            saved_labels=labels
                            first=False
                        else:
                            features=torch.cat((features,activation['layer4'].view(len(images),-1)))
                            saved_labels=torch.cat((saved_labels,labels))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 test images: %f %%' % (
                        100 * correct / float(total)))
                acc_train.append(100 * correct / float(total))

            
            
            y_test = np.asarray(saved_labels.cpu())
            tsne = TSNE().fit_transform(features.cpu())
            tx, ty = tsne[:,0], tsne[:,1]
            tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
            ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
            
            plt.figure(figsize = (16,12))


            for i in range(len(classes)):
                y_i = (y_test == i)

                plt.scatter(tx[y_i], ty[y_i], label=classes[i])
            plt.legend(loc=4)
            plt.gca().invert_yaxis()
            #plt.show()
            wandb_dict[args.mode+" t_sne"]=wandb.Image(plt)
            
            
            
            model.train()

        else:
            pass
            '''
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
            '''
        
        wandb_dict[args.mode + '_loss']= loss_avg
        #wandb_dict[args.mode + "_acc"]=acc_train[-1]
        wandb_dict['lr']=this_lr
        wandb.log(wandb_dict)

        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch == True:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch == True:
            this_alpha = args.alpha / (epoch + 1)




# In[ ]:




