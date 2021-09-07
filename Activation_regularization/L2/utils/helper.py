import models
import torch.optim as optim


__all__ = ['get_model', 'get_optimizer', 'get_scheduler']

def get_model(args):
    if args.set in ['CIFAR10',"MNIST"]:
        num_classes=10
    elif args.set in ["CIFAR100"]:
        num_classes=100
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes,l2_norm=args.l2_norm)
    return model


def get_optimizer(args, parameters):
    if args.set=='CIFAR10':
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.set=="MNIST":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.set=="CIFAR100":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        print("Invalid mode")
        return
    return optimizer



def get_scheduler(optimizer, args):
    if args.set=='CIFAR10':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: args.learning_rate_decay ** epoch,
                                )
    elif args.set=="MNIST":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
                                )
    elif args.set=="CIFAR100":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
                                )
    else:
        print("Invalid mode")
        return
    return scheduler