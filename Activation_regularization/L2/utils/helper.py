import models
import torch.optim as optim


__all__ = ['get_model', 'get_optimizer', 'get_scheduler']

def get_model(args):

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    return model


def get_optimizer(args, parameters):
    if args.set=='CIFAR10':
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.set=="MNIST":
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
    else:
        print("Invalid mode")
        return
    return scheduler