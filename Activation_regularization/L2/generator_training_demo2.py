
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
import models
import torchvision.utils as vutils
from PIL import Image
from utils import get_scheduler, get_optimizer, get_model, DeepInversionFeatureHook
from kornia import gaussian_blur2d
import torch.nn.functional as F


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_visible_device)

experiment_name=args.set+"_"+args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else "")+"_"+args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else '')
print(experiment_name)

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

generator = models.GeneratorBN(nz=3*32*32)

net = get_model(args)


## Load Pretrained Model
checkpoint_dir = '/data2/geeho/fed/CIFAR10/centralized/Fedavg'
checkpoint = torch.load(os.path.join(checkpoint_dir, 'Batch_norm_best.pth'))
print(net.load_state_dict(checkpoint['model_state_dict']))

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(generator.parameters(), lr=float(args.lr))
scheduler = get_scheduler(optimizer, args)
loss_train = []
acc_train=[]

kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

# preventing backpropagation through student for Adaptive DeepInversion
net = net.to(device)
generator = generator.to(device)
net.eval()



# set up criteria for optimization
criterion = nn.CrossEntropyLoss()
wandb_dict = {}

loss_r_feature_layers = []
for module in net.modules():
    if isinstance(module, nn.BatchNorm2d):
        loss_r_feature_layers.append(DeepInversionFeatureHook(module))

test_inputs = torch.randn((args.batch_size, 3, 32, 32), requires_grad=False, device=device)

## Training
for epoch in range(args.centralized_epochs):  # 데이터셋을 수차례 반복합니다.

    generator.train()
    running_loss = 0.0

    # initialize gaussian inputs
    inputs = torch.randn((args.batch_size, 3, 32, 32), requires_grad=True, device=device)
    inputs = inputs.to(device)
    optimizer.zero_grad()
    synthetic = generator(inputs)
    logits = net(synthetic)

    ## CE Loss
    pred = logits.data.max(1)[1]
    loss_one_hot = criterion(F.log_softmax(logits*args.g_temp), pred)

    ## Entropy Loss
    softmax_logits= torch.nn.functional.softmax(logits, dim=1).mean(dim=0)
    loss_information_entropy = (softmax_logits * torch.log10(softmax_logits)).sum()


    ##Stat Alignment Loss

    ## Create hooks for feature statistics catching
    rescale = [1. for _ in range(len(loss_r_feature_layers))]
    loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

    ## Smoothness Prior Loss

    ## Get blurred image by gaussian kernel
    blurred_synthetic = gaussian_blur2d(synthetic.detach(), (3, 3), (1.5, 1.5))

    prior_loss = ((synthetic - blurred_synthetic) ** 2).sum()

    total_loss = args.g1 * loss_one_hot + args.g2 * loss_information_entropy + args.g3 * prior_loss + 5e1 * loss_r_feature

    total_loss.backward()
    optimizer.step()
    #scheduler.step()

    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            generator.eval()

            test_inputs = test_inputs.to(device)
            synthetic = generator(inputs)

            logits = net(synthetic)
            pred = logits.data.max(1)[1]

            for id in range(synthetic.size(0)):
                image_np = synthetic[id].data.cpu().numpy().transpose((1, 2, 0))
                pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
                wandb_dict['synthetic_image_%d'%id] = [wandb.Image(pil_image, caption="global_epoch" + str(epoch+1) + " predicted_class:" + str(pred[id].item()))]


            wandb_dict['input_image'] = [wandb.Image(test_inputs, caption="global_epoch" + str(epoch + 1))]
            wandb.log(wandb_dict)
            print('image added!')




# ## TEST
# correct = 0
# total = 0
# with torch.no_grad():
#     generator.eval()
#     inputs = torch.randn((args.batch_size, 3, 32, 32), requires_grad=True, device=device)
#     inputs = inputs.to(device)
#
#     synthetic = generator(inputs)
#
#     wandb_dict['common_image'] = [wandb.Image(synthetic, caption="global_epoch" + str(epoch))]
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %f %%' % (
#     100 * correct / total))

