#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


from args import args


# In[3]:


# In[4]:



import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import copy
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_visible_device)

import matplotlib.pyplot as plt
import torch.optim as optim
import random
import models
import json
import wandb


# In[6]:


experiment_name=args.set+"_"+args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else "")+"_"+args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else '')
print(experiment_name)  


# In[7]:


wandb.init(entity='federated_learning',project=args.project,group=args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else ""),job_type=args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else ''))
wandb.run.name=experiment_name
wandb.run.save()
wandb.config.update(args)




# In[8]:


random_seed=args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# In[9]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:





# In[10]:


if args.set=='CIFAR10':
    
    
    
    transform_train = transforms.Compose(
        [transforms.RandomRotation(10),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))])    
    transform_test = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))])    
    
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
elif args.set=='MNIST':
    #!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    #!tar -zxvf MNIST.tar.gz

    transform = transforms.Compose(
        [
         #transforms.RandomHorizontalFlip(),
         #transforms.RandomVerticalFlip(),
         #transforms.RandomCrop(28, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))
         ])   
    trainset = datasets.MNIST(root=args.data, train=True,
                                            download=True,
                                   transform=transform)
    
    trainloader=torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)    
    testset=datasets.MNIST(root=args.data, train=False,
                                           download=True,
                                   transform=transform)

    testloader=torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)


# In[11]:


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[12]:


# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #print(npimg)
    print(np.transpose(npimg, (1, 2, 0)).shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(np.transpose(npimg))#, (1, 2, 0)))
    plt.show()


# In[13]:


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# In[14]:


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    #num_items=8
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# In[15]:


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    #num_shards, num_imgs = 200, 250
    #num_shards, num_imgs = 200, 250
    class_per_user=1
    num_shards=num_users*class_per_user
    num_imgs=int(len(dataset)/num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([],dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    #labels = np.array(dataset.train_labels)

    labels=[]
    for element in dataset:
        labels.append(int(element[1]))
    #print(type(labels[0]))
    labels=np.array(labels)
    #labels=labels.astype('int64')
    # sort labels
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, class_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = set(np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0))
    return dict_users


# In[16]:


def cifar_dirichlet(dataset, n_nets, alpha=0.5):
    '''
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    '''
    #X_train=dataset[:][0]
    y_train=torch.zeros(len(dataset),dtype=torch.long)
    print(y_train.dtype)
    for a in range(len(dataset)):
        y_train[a]=(dataset[a][1])
    n_train = len(dataset)
    #X_train.shape[0]
    '''
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    '''
    #elif partition == "hetero-dir":
    min_size = 0
    K = 10
    N=len(dataset)
    N = y_train.shape[0]
    net_dataidx_map = {i: np.array([],dtype='int64') for i in range(n_nets)}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    #traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return net_dataidx_map
    #return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


# In[17]:


def get_model(args):

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    return model


# In[18]:


def get_optimizer():
    if args.set=='CIFAR10':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.set=="MNIST":
        optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        print("Invalid mode")
        return
    return optimizer


# In[19]:


def get_scheduler(optimizer):
    if args.set=='CIFAR10':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: args.learning_rate_decay ** epoch,
                                )
    elif args.set=="MNIST":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
                                )
    else:
        print("Invalid mode")
        return
    return scheduler


# In[20]:


if args.mode=='centralized':
    net = get_model(args)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer()
    scheduler = get_scheduler(optimizer)
    loss_train = []
    acc_train=[]


# In[ ]:





# In[21]:


if args.mode=='centralized':
    for epoch in range(args.centralized_epochs):   # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data[0].to(device), data[1].to(device)
            #print(labels)

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            #print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            


        if epoch%args.print_freq==0:
            loss_train.append(loss)
            print(f"epoch: {epoch}")
            print(' Average loss {:.3f}'.format( loss))
            for j in range(1):
                net.eval()         
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / total))
            acc_train.append(100 * correct / total)

            net.train()
        scheduler.step()
            


# In[22]:


if args.mode=='centralized':
    fig,ax1=plt.subplots()
    ax2=ax1.twinx()
    line1=ax1.plot(np.array(loss_train),color='g',label='loss_train')
    line2=ax2.plot([i*1 for i in range(len(acc_train))],acc_train,label='acc_train')
    lines=line1+line2
    ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy")
    plt.xlabel('Epoch')
    plt.title('Experiment Result')
    plt.legend(lines,['loss_train','acc_train'])
    plt.show()


# In[23]:


if args.mode=='centralized':
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))


# In[24]:


class LocalUpdate(object):
    def __init__(self, lr,local_epoch,device,batch_size, dataset=None, idxs=None,alpha=0.0001):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha

    def train(self, net):
        net.sync_online_and_global()
        net.train()
        # train and update
        max_norm = 5
        optimizer = optim.SGD(net.parameters(), lr=self.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs,activation_l2 = net(images,online_target=True)
                loss = self.loss_func(log_probs, labels)+self.alpha*activation_l2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# In[25]:


def get_dataset(mode='iid'):
    directory=args.client_data+'/'+args.set+'/'+mode+(str(args.dirichlet_alpha) if mode=='dirichlet' else '')+'.txt'
    check_already_exist=os.path.isfile(directory) and (os.stat(directory).st_size != 0)
    create_new_client_data=not check_already_exist or args.create_client_dataset
    print("create new client data: "+str(create_new_client_data))
    
    if create_new_client_data==False:
        try:
            dataset={}
            with open(directory) as f:
                for idx,line in enumerate(f):
                    dataset=eval(line)
        except:
            print("Have problem to read client data")
        
    
    if create_new_client_data==True:
        if mode=='iid':
            dataset=cifar_iid(trainset, args.num_of_clients)
        elif mode=='skew1class':
            dataset=cifar_noniid(trainset, args.num_of_clients)
        elif mode=='dirichlet':
            dataset=cifar_dirichlet(trainset, args.num_of_clients,alpha=args.dirichlet_alpha)
        else:
            print("Invalid mode ==> please select in iid, skew1class, dirichlet")
            return
        try:
            os.makedirs(args.client_data+'/'+args.set,exist_ok=True)
            with open(directory, 'w') as f:
                print(dataset, file=f)
            
        except:
            print("Fail to write client data at "+directory)
        
    return dataset
    


# In[26]:


def do_federated_learning(mode='iid'):
    FedAvg_model=get_model(args)
    FedAvg_model.to(device)
    wandb.watch(FedAvg_model)
    criterion= nn.CrossEntropyLoss().to(device)
    criterion= nn.CrossEntropyLoss().to(device)
    FedAvg_model.train()
    epoch_loss = []
    weight_saved=FedAvg_model.state_dict()
    
    dataset=get_dataset(mode)
    loss_train = []
    acc_train=[]
    this_lr=args.lr
    this_alpha=args.alpha
    
    
    for epoch in range(args.global_epochs):
        local_weight=[]
        local_loss=[]
        m=max(int(args.participation_rate*args.num_of_clients),1)
        selected_user=np.random.choice(range(args.num_of_clients),m,replace=False)
        print(f"This is global {epoch} epoch")
        for user in selected_user:
            local_setting=LocalUpdate(lr=this_lr,local_epoch=args.local_epochs,device=device,batch_size=args.batch_size,dataset=trainset,idxs=dataset[user],alpha=this_alpha)
            weight,loss=local_setting.train(net=copy.deepcopy(FedAvg_model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
        FedAvg_weight=copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(1,len(local_weight)):
                FedAvg_weight[key]+=local_weight[i][key]
            FedAvg_weight[key]/=len(local_weight)
        FedAvg_model.load_state_dict(FedAvg_weight)
        loss_avg = sum(local_loss) / len(local_loss)
        print(' Average loss {:.3f}'.format( loss_avg))
        loss_train.append(loss_avg)
        if epoch%args.print_freq==0:
            FedAvg_model.eval()         
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = FedAvg_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                100 * correct / float(total)))
            acc_train.append(100 * correct / float(total))

        FedAvg_model.train()
        
        
        wandb.log({mode+'_loss':loss_avg,mode+"_acc":acc_train[-1]})
        
        this_lr*=args.learning_rate_decay
        if args.alpha_mul_epoch==True:
            this_alpha=args.alpha*(epoch+1)
        elif args.alpha_divide_epoch==True:
            this_alpha=args.alpha/(epoch+1)
            
    fig,ax1=plt.subplots()
    ax2=ax1.twinx()
    line1=ax1.plot(np.array(loss_train),color='g',label='loss_train')
    line2=ax2.plot([i*1 for i in range(len(acc_train))],acc_train,label='acc_train')
    lines=line1+line2
    ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy")
    plt.xlabel('Epoch')
    plt.title('Experiment Result')
    plt.legend(lines,['loss_train','acc_train'])
    plt.show()
    
    
    print('loss_train')
    print(loss_train)
    
    print('acc_train')
    print(acc_train)
    


# In[27]:


do_federated_learning(args.mode)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# cifar10 FedAvg iid로 학습

# if args.federated_iid==True:
#     do_federated_learning(mode)

# cifar10 FedAvg Non-iid(skew1class)로 학습
# 

# if args.federated_skew1class==True:
#     do_federated_learning(mode='skew1class')

# cifar10 FedAvg Non-iid(dirichlet)로 학습

# if args.federated_dirichlet==True:
#     do_federated_learning(mode='dirichlet')

# In[ ]:





# In[ ]:




