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

def create_random_image_criterion(size,class_num):
    x=torch.tensor(0.006).repeat(size,10)
    x[:,class_num]=7
    x+=torch.normal(mean=0,std=0.0045,size=(size,10))
    x=x
    return torch.softmax(x,dim=1)



def filter_visualize(image_epochs,net,device,size=1):
    with torch.no_grad():
        for a in net.parameters():
            a.requires_grad=False
    
    image=(torch.zeros([10,3,32,32],requires_grad=True,device=device))
    for class_num in range(10):
        if class_num==0:
            image_criterion=create_random_image_criterion(size,class_num)
        else:
            image_criterion=torch.cat((image_criterion,create_random_image_criterion(size,class_num)))
                                      
    image_criterion=image_criterion.to(device)        
    image_optimizer=torch.optim.SGD([image],lr=0.1,momentum=0.5)

    for i in range(image_epochs):
        image_optimizer.zero_grad()
        output=net(image)
        loss=((torch.softmax(output,dim=1)-image_criterion)**2).sum()+((image**2).sum())*0.001
        loss.backward()
        image_optimizer.step()

    for a in net.parameters():
        a.requires_grad=True
    image.requires_grad=False
    return image



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
    

    for epoch in range(args.global_epochs):
        wandb_dict={}
        num_of_data_clients=[]
        local_weight = []
        local_loss = []
        m = max(int(args.participation_rate * args.num_of_clients), 1)
        selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        print(f"This is global {epoch} epoch")
        
        
        
        idea_img=filter_visualize(args.image_epochs,model,device,size=1)
        idea_label=torch.range(start=0,end=9,dtype=int).to(device)
        
        #idea_img+=torch.normal(mean=0,std=0.1,size=idea_img.shape).to(device)
        #idea_img.clamp(0,1)
        model.eval()
        log_probs=torch.softmax(model(idea_img),dim=1)
        print(log_probs)
        model.train()
        wandb_dict['common_image']=[wandb.Image(idea_img, caption="global_epoch"+str(epoch))]
        wandb.log({f"log probs-{i}": el[i] for i, el in enumerate(log_probs)},commit=False)
        
        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                        batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device), idea_img=idea_img,idea_label=idea_label)
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
                                      
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

            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            for i in range(len(classes)):
                y_i = (y_test == i)

                plt.scatter(tx[y_i], ty[y_i], label=classes[i])
            plt.legend(loc=4)
            plt.gca().invert_yaxis()
            #plt.show()
            wandb_dict[args.mode+" t_sne"]=wandb.Image(plt)
            
            
            
            model.train()
        else:
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




