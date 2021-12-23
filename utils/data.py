import torch
from torchvision import datasets, transforms
from torch.utils.data import  Dataset
import os
from datasets.cifar import cifar_noniid, cifar_dirichlet_balanced,cifar_dirichlet_unbalanced, cifar_iid
import torch.nn as nn


__all__ = ['DatasetSplit', 'DatasetSplitMultiView', 'get_dataset']

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

class DatasetSplitMultiView(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (view1, view2), label = self.dataset[self.idxs[item]]
        return torch.tensor(view1), torch.tensor(view2), torch.tensor(label)


def get_dataset(args, trainset, mode='iid'):
    directory = args.client_data + '/' + args.set + '/' + ('un' if args.data_unbalanced==True else '') + 'balanced'
    filepath=directory+'/' + mode + (str(args.dirichlet_alpha) if mode == 'dirichlet' else '') + '_clients' +str(args.num_of_clients) +'.txt'
    check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
    create_new_client_data = not check_already_exist or args.create_client_dataset
    print("create new client data: " + str(create_new_client_data))

    if create_new_client_data == False:
        try:

            dataset = {}
            with open(filepath) as f:
                for idx, line in enumerate(f):
                    dataset = eval(line)
        except:
            print("Have problem to read client data")

    if create_new_client_data == True:
        if mode == 'iid':
            dataset = cifar_iid(trainset, args.num_of_clients)
        elif mode == 'skew1class':
            dataset = cifar_noniid(trainset, args.num_of_clients)
        elif mode == 'dirichlet':
            if args.data_unbalanced==True:
                dataset = cifar_dirichlet_unbalanced(trainset, args.num_of_clients, alpha=args.dirichlet_alpha)
            else:
                dataset = cifar_dirichlet_balanced(trainset, args.num_of_clients, alpha=args.dirichlet_alpha)
        else:
            print("Invalid mode ==> please select in iid, skew1class, dirichlet")
            return
        try:
            os.makedirs(directory, exist_ok=True)
            with open(filepath, 'w') as f:
                print(dataset, file=f)

        except:
            print("Fail to write client data at " + directory)

    return dataset

