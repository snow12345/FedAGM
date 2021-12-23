import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit
import torch
from local_update_method.global_and_online_model import *


####### batch_size는 client batch_size의 num_of_participation_clients배가 되어야 할 것.
class CenterUpdate(object):
    def __init__(self, args, lr, iteration_num, device, batch_size, dataset=None, idxs=None,num_of_participation_clients = 10):
        self.lr=lr
        self.iteration_num=iteration_num
        self.device=device
        self.loss_func=nn.CrossEntropyLoss()
        self.selected_clients = []

        if idxs ==None:
            self.idxs  = range(len(dataset))

        else:
            self.idxs=idxs           
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs), batch_size=batch_size, shuffle=True)
        self.args=args
        self.K = len(self.ldr_train)

    def train(self, net, delta=None):
        model = net
        optimizer = optim.SGD(model.parameters(), lr=self.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        count = 0
        while(True):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if count ==self.iteration_num:
                    break
                count+=1
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
            if count ==self.iteration_num:
                break 
        return net.state_dict()
