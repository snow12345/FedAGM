import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplitMultiView, sigmoid_rampup
import torch
from local_update_method.global_and_online_model import *

class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        self.loss_func = nn.CrossEntropyLoss()
        #self.loss_func = nn.NLLLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplitMultiView(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args
        self.K = len(self.ldr_train)

    def train(self, net, predictor, delta=None, epoch=0):
        #model=dual_model(self.args,net,net)
        target_network = copy.deepcopy(net)
        for param_t in target_network.parameters():
            param_t.requires_grad = False
        model = net
        predictor = predictor
        # train and update
        w = self.args.rampup_coefficient * sigmoid_rampup(epoch, self.args.rampup_length)
        
        optimizer = optim.SGD(list(model.parameters()) + list(predictor.parameters()), lr=self.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (view1, view2, labels) in enumerate(self.ldr_train):
                view1, view2, labels = view1.to(self.device), view2.to(self.device), labels.to(self.device)
                net.zero_grad()
                predictor.zero_grad()
                out1, logit1 = model(view1)
                out2, logit2 = model(view2)
                #log_probs1 = F.log_softmax(logit1/self.args.temp, dim=1)
                #log_probs2 = F.log_softmax(logit2 / self.args.temp, dim=1)
                loss = self.loss_func(logit1, labels)
                loss += self.loss_func(logit2, labels)

                ## regression loss

                predictions1 = predictor(out1)
                predictions2 = predictor(out2)

                with torch.no_grad():
                    targets_to_view2, _ = target_network(view1)
                    targets_to_view1, _ = target_network(view2)

                consistency_loss = self.regression_loss(predictions1, targets_to_view1)
                consistency_loss += self.regression_loss(predictions2, targets_to_view2)

                loss = loss + w * consistency_loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), predictor.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def initializes_target_network(self, online_network, target_network):
        # init momentum network as encoder net
        for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def regression_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
