import torch
from torchvision import datasets, transforms
from torch.utils.data import  Dataset
import os
from datasets.cifar import cifar_noniid, cifar_dirichlet, cifar_iid


__all__ = ['DatasetSplit', 'get_dataset']


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


def get_dataset(args, mode='iid'):
    directory = args.client_data + '/' + args.set + '/' + mode + (
        str(args.dirichlet_alpha) if mode == 'dirichlet' else '') + '.txt'
    check_already_exist = os.path.isfile(directory) and (os.stat(directory).st_size != 0)
    create_new_client_data = not check_already_exist or args.create_client_dataset
    print("create new client data: " + str(create_new_client_data))

    if create_new_client_data == False:
        try:
            dataset = {}
            with open(directory) as f:
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
            dataset = cifar_dirichlet(trainset, args.num_of_clients, alpha=args.alpha)
        else:
            print("Invalid mode ==> please select in iid, skew1class, dirichlet")
            return
        try:
            os.makedirs(args.client_data + '/' + args.set, exist_ok=True)
            with open(directory, 'w') as f:
                print(dataset, file=f)

        except:
            print("Fail to write client data at " + directory)

    return dataset