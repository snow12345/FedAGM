import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit
import torch
from local_update_method.global_and_online_model import *



def KD(input_p,input_q,T=1):
    p=F.softmax((input_p/T),dim=1)
    q=F.softmax((input_q/T),dim=1)
    result=((p*((p/q).log())).sum())/len(input_p)
    
    if not torch.isfinite(result):
        print('==================================================================')
        print('input_p')
        print(input_p)
        
        print('==================================================================')
        print('input_q')
        print(input_q)
        print('==================================================================')
        print('p')
        print(p)
        
        print('==================================================================')
        print('q')
        print(q)
        
        
        print('******************************************************************')
        print('p/q')
        print(p/q)
        
        print('******************************************************************')
        print('(p/q).log()')
        print((p/q).log())        
        
        print('******************************************************************')
        print('(p*((p/q).log())).sum()')
        print((p*((p/q).log())).sum())            
    
    return result



class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        #self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = nn.NLLLoss()
        self.loss_func2=nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args
        self.K = len(self.ldr_train)

    def train(self, net, idea_img,idea_label,delta=None):
        #model = net
        model=dual_model(self.args,net,net)
        net.l2_norm = True
        fixed_model = copy.deepcopy(net)
        # train and update
        
        optimizer = optim.SGD(model.parameters(), lr=self.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                images=torch.cat((images,idea_img),dim=0)
                
                net.zero_grad()
                logits,activation_loss = model(images,online_target=True)
                log_probs = F.log_softmax(logits[:-10]/self.args.temp, dim=1)
                loss = self.loss_func(log_probs, labels)+self.alpha*activation_loss+self.loss_func2(idea_label,logits[-10:])

                ## Weight L2 loss
                reg_loss = 0
                fixed_params = {n: p for n, p in fixed_model.named_parameters()}
                for n, p in net.named_parameters():
                    if 'linear' in n:
                        reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()
                loss = loss + 0.5 * self.args.mu * reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
