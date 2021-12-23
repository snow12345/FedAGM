import torch

def check_data_distribution(dataloader):
    data_distribution=torch.zeros(10)
    for idx,(images,target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def check_data_distribution_aug(dataloader):
    data_distribution=torch.zeros(10)
    for idx,(images, _, target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution
