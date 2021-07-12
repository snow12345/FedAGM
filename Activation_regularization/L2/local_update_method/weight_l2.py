import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit
import torch
import copy


class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None,alpha=0.0):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args

    def train(self, net):
        net.sync_online_and_global()
        net.train()
        fixed_model = copy.deepcopy(net)
        # train and update
        max_norm = 5
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = net(images,online_target=True)
                loss = self.loss_func(log_probs, labels)

                ## Weight L2 loss

                reg_loss = 0
                fixed_params = {n:p for n,p in fixed_model.named_parameters()}
                for n, p in net.named_parameters():
                    reg_loss += ((p-fixed_params[n].detach())**2).sum()
                loss = loss + 0.5 * self.args.mu * reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)