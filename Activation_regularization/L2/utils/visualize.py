import matplotlib.pyplot as plt
import numpy as np
import torch
from global_update_method.distcheck import check_data_distribution
import umap.umap_ as umap
from mpl_toolkits import mplot3d
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from torch.utils.data import DataLoader
import wandb


__all__ = ['imshow', 'log_acc', 'log_ConfusionMatrix_Umap', 'get_activation']

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #print(npimg)
    print(np.transpose(npimg, (1, 2, 0)).shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(np.transpose(npimg))#, (1, 2, 0)))
    plt.show()


def log_acc(model, testloader, args, wandb_dict, name):
    model.eval()
    device = next(model.parameters()).device
    first = True
    with torch.no_grad():
        for data in testloader:
            activation = {}
            model.layer4.register_forward_hook(get_activation('layer4', activation))
            images, labels = data[0].to(device), data[1].to(device)
            if 'byol' in args.method:
                _ ,outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if first:
                features = activation['layer4'].view(len(images), -1)
                saved_labels = labels
                saved_pred = predicted
                first = False
            else:
                features = torch.cat((features, activation['layer4'].view(len(images), -1)))
                saved_labels = torch.cat((saved_labels, labels))
                saved_pred = torch.cat((saved_pred, predicted))

        saved_labels = saved_labels.cpu()
        saved_pred = saved_pred.cpu()

        f1 = metrics.f1_score(saved_labels, saved_pred, average='weighted')
        acc = metrics.accuracy_score(saved_labels, saved_pred)
        wandb_dict[name + " f1"] = f1
        wandb_dict[name + " acc"] = acc

    model.train()
    return acc


def log_ConfusionMatrix_Umap(model, testloader, args, wandb_dict, name):
    if args.set == 'CIFAR10':
        classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.set == 'MNIST':
        classes=['0','1','2','3','4','5','6','7','8','9']
    else:
        pass
    
    
    
    
    model.eval()
    device = next(model.parameters()).device
    first = True
    with torch.no_grad():
        for data in testloader:
            activation = {}
            model.layer4.register_forward_hook(get_activation('layer4', activation))
            images, labels = data[0].to(device), data[1].to(device)
            if 'byol' in args.method:
                _, outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if first:
                features = activation['layer4'].view(len(images), -1)
                saved_labels = labels
                saved_pred = predicted
                first = False
            else:
                features = torch.cat((features, activation['layer4'].view(len(images), -1)))
                saved_labels = torch.cat((saved_labels, labels))
                saved_pred = torch.cat((saved_pred, predicted))

        saved_labels = saved_labels.cpu()
        saved_pred = saved_pred.cpu()

        # plt.figure()
        f1 = metrics.f1_score(saved_labels, saved_pred, average='weighted')
        acc = metrics.accuracy_score(saved_labels, saved_pred)
        cm = metrics.confusion_matrix(saved_labels, saved_pred)
        wandb_dict[name + " f1"] = f1
        wandb_dict[name + " acc"] = acc
        plt.figure(figsize=(20, 20))
        # wandb_dict[args.mode+name+" f1"]=f1
        # wandb_dict[args.mode+name+" acc"]=acc
        fig, ax = plot_confusion_matrix(cm, class_names=classes,
                                        colorbar=True,
                                        show_absolute=False,
                                        show_normed=True,
                                        figsize=(16, 16)
                                        )
        ax.margins(2, 2)

        wandb_dict[name + " confusion_matrix"] = wandb.Image(fig)
        plt.close()
        y_test = np.asarray(saved_labels.cpu())

        reducer = umap.UMAP(random_state=0, n_components=args.umap_dim)
        embedding = reducer.fit_transform(features.cpu())
        
        
        ##################### plot ground truth #######################
        plt.figure(figsize=(20, 20))

        if args.umap_dim == 3:
            ax = plt.axes(projection=('3d'))
        else:
            ax = plt.axes()

        for i in range(len(classes)):
            y_i = (y_test == i)
            scatter_input = [embedding[y_i, k] for k in range(args.umap_dim)]
            ax.scatter(*scatter_input, label=classes[i])
        plt.legend(loc=4)
        plt.gca().invert_yaxis()

        wandb_dict[name + " umap"] = wandb.Image(plt)
        plt.close()
        
        
        
        ############### plot model predicted class ###########################
        plt.figure(figsize=(20, 20))

        if args.umap_dim == 3:
            ax = plt.axes(projection=('3d'))
        else:
            ax = plt.axes()

        for i in range(len(classes)):
            y_i =(np.asarray(saved_pred.cpu()) == i)
            scatter_input = [embedding[y_i, k] for k in range(args.umap_dim)]
            ax.scatter(*scatter_input, label=classes[i])
        plt.legend(loc=4)
        plt.gca().invert_yaxis()

        wandb_dict[name + " umap_model predicted class"] = wandb.Image(plt)
        plt.close()        
        
        
    model.train()
    return acc


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook