import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit, FedCM_SGD
import torch
import copy


class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args
        self.K = len(self.ldr_train)

    def train(self, net, delta=None):
        #net.sync_online_and_global()
        net.train()
        fixed_model = copy.deepcopy(net)
        for param_t in fixed_model.parameters():
            param_t.requires_grad = False
        # train and update

        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                if self.args.arch == "ResNet18":
                    log_probs = net(images)
                else:
                    log_probs= net(images)
                ce_loss = self.loss_func(log_probs, labels)


                # ## Weight L2 loss
                # reg_loss = 0
                # fixed_params = {n: p for n, p in fixed_model.named_parameters()}
                # for n, p in net.named_parameters():
                #     reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()

                #loss = self.args.alpha * ce_loss + 0.5 * self.args.mu * reg_loss
                loss = self.args.alpha * ce_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def calculate_importance(self, dataloader, model):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        print('Computing EWC')

        # Initialize the importance matrix
        params = {n: p for n, p in model.named_parameters()}
        importance = {}
        for n, p in params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        model.eval()

        # Accumulate the square of gradients
        for i, (input, target) in enumerate(dataloader):

            input = input.to(self.device)
            target = target.to(self.device)

            preds = self.forward(input)


            # The flag self.valid_out_dim is for handling the case of incremental class learning.
            # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
            # in calculating the loss.
            pred = preds[task_name] if not isinstance(self.valid_out_dim, int) else preds[task_name][:,
                                                                                    :self.valid_out_dim]
            ind = pred.max(1)[1].flatten()  # Choose the one with max

            # - Alternative ind by multinomial sampling. Its performance is similar. -
            # prob = torch.nn.functional.softmax(preds['All'],dim=1)
            # ind = torch.multinomial(prob,1).flatten()

            if self.empFI:  # Use groundtruth label (default is without this)
                ind = target

            loss = self.criterion(preds, ind, task, regularization=False)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)

        return importance
