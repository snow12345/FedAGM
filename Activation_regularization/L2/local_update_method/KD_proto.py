import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
import torch
from local_update_method.global_and_online_model import *
import copy




class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        #self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = nn.NLLLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args
        self.K = len(self.ldr_train)
        self.num_classes=get_numclasses(args)
    def train(self, net, idea_img,idea_label,delta=None):
        
        # calculate data distribution and set loss function with weight based on data distribution
        synthesize_dist=count_label_distribution(idea_label,class_num=self.num_classes)
        total_dist=check_data_distribution(self.ldr_train,class_num=self.num_classes,default_dist=synthesize_dist*len(idea_label))
        class_weight=1/total_dist
        class_weight=class_weight.to(self.device)
        self.weighted_loss_func=nn.NLLLoss(weight=class_weight)
        
        
        
        #set global model and online model
        model=net

        global_model = copy.deepcopy(net)
        for par in global_model.parameters():
            par.requires_grad=False
        
        optimizer = optim.SGD(model.parameters(), lr=self.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        epoch_loss = []
        for epoch in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                num_of_realdata_in_batch=len(labels)                
                images=torch.cat((images,idea_img),dim=0)
                labels=torch.cat((labels,idea_label),dim=0)
                optimizer.zero_grad()
                local_feature,local_logit = model(images,return_feature=True)
                global_feature,global_logit = global_model(images,return_feature=True)
                detached_local_feature=local_feature.detach().clone()
                
                
                
                ####### (12) L_CE ########################
                log_probs = F.log_softmax(local_logit[:num_of_realdata_in_batch]/self.args.temp, dim=1)
                L_CE=self.loss_func(log_probs, labels[:num_of_realdata_in_batch])

                
                
                ######  (13) L_KD ########################
                if (batch_idx+epoch)!=0:
                    w_local_feature=global_model.forward_classifier(local_feature)
                    L_KD=l2norm(w_local_feature,global_logit)
                else:
                    L_KD=0
                
                ######  (14) L_FT ########################
                local_logit_FT=model.forward_classifier(detached_local_feature)
                log_probs_FT = F.log_softmax(local_logit_FT/self.args.temp, dim=1)
                L_FT=self.weighted_loss_func(log_probs_FT, labels)
                
                
                loss=L_CE+self.args.alpha*L_KD+self.args.mu*L_FT
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
