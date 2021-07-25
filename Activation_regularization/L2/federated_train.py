
import sys
from args_dir.federated import args
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import os
import random
import wandb
from build_method import build_local_update_module
from build_global_method import build_global_update_module
from utils import MultiViewDataInjector, GaussianBlur
from utils import get_scheduler, get_optimizer, get_model, get_dataset
import copy



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_visible_device)

experiment_name=args.set+"_"+args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else "")+"_"+args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else '')
print(experiment_name)

wandb.init(entity='federated_learning', project=args.project,group=args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else ""),job_type=args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else ''))
wandb.run.name=experiment_name
wandb.run.save()
wandb.config.update(args)

random_seed=args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Build Dataset
if args.set == 'CIFAR10':

    if args.method != 'byol':
        transform_train = transforms.Compose(
            [transforms.RandomRotation(10),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

        trainset = torchvision.datasets.CIFAR10(root=args.data, train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR10(root=args.data, train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.workers)
    else:
        color_jitter = transforms.ColorJitter(0.4 * 1, 0.4 * 1, 0.4 * 1, 0.1 * 1)
        transform_train = transforms.Compose(
            [transforms.RandomRotation(10),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.RandomApply([color_jitter], p=0.8),
             transforms.RandomGrayscale(p=0.2),
             #GaussianBlur(kernel_size=int(0.1 * 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

        trainset = torchvision.datasets.CIFAR10(root=args.data, train=True,
                                                download=True, transform=MultiViewDataInjector([transform_train, transform_train]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR10(root=args.data, train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.workers)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

elif args.set == 'MNIST':
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    trainset = datasets.MNIST(root=args.data, train=True,
                              download=True,
                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)
    testset = datasets.MNIST(root=args.data, train=False,
                             download=True,
                             transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers)



LocalUpdate = build_local_update_module(args)
global_update=build_global_update_module(args)
global_update(args=args,device=device,trainset=trainset,testloader=testloader,LocalUpdate=LocalUpdate)

'''
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
    local_weight = []
    local_loss = []
    m = max(int(args.participation_rate * args.num_of_clients), 1)
    selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
    print(f"This is global {epoch} epoch")
    for user in selected_user:
        local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                    batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
        weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
        local_weight.append(copy.deepcopy(weight))
        local_loss.append(copy.deepcopy(loss))
    FedAvg_weight = copy.deepcopy(local_weight[0])
    for key in FedAvg_weight.keys():
        for i in range(1, len(local_weight)):
            FedAvg_weight[key] += local_weight[i][key]
        FedAvg_weight[key] /= len(local_weight)
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
                outputs = model(images)
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
'''
