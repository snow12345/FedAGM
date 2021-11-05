
import sys
from args_dir.centralized import args
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import os
import random
import wandb

from utils import get_scheduler, get_optimizer, get_model


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_visible_device)

experiment_name=args.set+"_"+args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else "")+"_"+args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else '')
print(experiment_name)

LOG_DIR = '/data/private/geeho/fed/{}/{}/{}'.format(args.set,'centralized', args.method)
if not os.path.exists('{}'.format(LOG_DIR)):
    os.makedirs('{}'.format(LOG_DIR))

wandb.init(entity='federated_learning', project=args.project, group="centralized",job_type=args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else ''))
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
    
    if args.method not in ['byol','simsiam']:
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
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

elif args.set == 'CIFAR100':
    if args.method not in ['byol', 'simsiam']:
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
             # GaussianBlur(kernel_size=int(0.1 * 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    trainset = torchvision.datasets.CIFAR100(root=args.data, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR100(root=args.data, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)

elif args.set == 'MNIST':
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz

    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomCrop(28, padding=4),
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



net = get_model(args)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(args, net.parameters())
scheduler = get_scheduler(optimizer, args)
loss_train = []
acc_train=[]
best_acc = 0

## Training
for epoch in range(args.centralized_epochs):  # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data[0].to(device), data[1].to(device)
        # print(labels)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if epoch % args.print_freq == 0:
        loss_train.append(loss)
        print(f"epoch: {epoch}")
        print(' Average loss {:.3f}'.format(loss))
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
        acc = 100 * correct / total
        if best_acc < 100 * correct / total:
            best_acc = 100 * correct / total

            print('Best Accuracy of the network on the 10000 test images: %f %%' % (
                    best_acc))

            torch.save({'model_state_dict': net.state_dict()},
                       '{}/{}_{}.pth'.format(LOG_DIR, args.additional_experiment_name if args.additional_experiment_name!='' else '', 'best')
                       )
        wandb.log({
            "acc": acc,
            'best_acc': best_acc,
        })

        acc_train.append(100 * correct / total)

        net.train()
    scheduler.step()



## TEST
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

