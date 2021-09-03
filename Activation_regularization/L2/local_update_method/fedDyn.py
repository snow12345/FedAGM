import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit
import torch
from local_update_method.global_and_online_model import *
from tqdm import *

class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0, local_deltas=None):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args
        self.K = len(self.ldr_train)
        self.local_deltas = local_deltas

    def train(self, net, delta=None, user=None):
        #model=dual_model(self.args,net,net)
        model = net
        fixed_model = copy.deepcopy(net)
        fixed_params = {n: p for n, p in fixed_model.named_parameters()}


        # train and update
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        epoch_loss = []
        local_delta = copy.deepcopy(self.local_deltas[user])
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = model(images)
                ce_loss = self.loss_func(log_probs, labels)

                ## Weight L2 loss
                reg_loss = 0
                for n, p in net.named_parameters():
                    reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()


                ## local gradient regularization
                lg_loss = 0
                for n, p in net.named_parameters():
                    p = torch.flatten(p)
                    local_d = local_delta[n].detach().clone().to(self.device)
                    local_grad = torch.flatten(local_d)
                    lg_loss += (p * local_grad.detach()).sum()

                #loss = ce_loss - lg_loss + 0.5 * self.args.mu * reg_loss
                loss = ce_loss - lg_loss + 0.5 * self.args.mu * reg_loss
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #model.to('cpu')

        ## Update Local Delta
        for n, p in net.named_parameters():
            self.local_deltas[user][n] = (local_delta[n] - self.args.alpha * (p - fixed_params[n]).detach().clone().to('cpu'))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
