import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit,IL
import torch
from local_update_method.global_and_online_model import *

class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        if args.loss=='CE':
            self.loss_func=nn.CrossEntropyLoss()
        elif args.loss in ('IL','Individual_loss'):
            self.loss_func=IL(device=device)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args
        self.K = len(self.ldr_train)

    def train(self, net, delta=None):
        #model=dual_model(self.args,net,net)
        model = net
        # train and update
        
        init=copy.deepcopy(model.state_dict())
        mask=copy.deepcopy(init)
        tmp=copy.deepcopy(init)
        
        
        optimizer = optim.SGD(model.parameters(), lr=self.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if (iter!=0) and (batch_idx!=0):
                    before=copy.deepcopy(model.state_dict())
                    for key in before:
                        mask[key]=torch.randint(0,2,init[key].shape).to(self.device)
                        tmp[key]=mask[key]*(1.1*init[key]-0.1*before[key])+2.1*(1-mask[key])*before[key]
                    model.load_state_dict(tmp)
                
                
                
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
                
                
                if (iter!=0) and (batch_idx!=0):
                    after=copy.deepcopy(model.state_dict())
                    for key in after:
                        tmp[key]=mask[key]*before[key]+(1-mask[key])*after[key]
                    model.load_state_dict(tmp)
                
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
