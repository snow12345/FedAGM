import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit
import torch
#from federated_train import transform_train
from local_update_method.global_and_online_model import *
from torchvision import transforms
transform= transforms.Compose(
            [transforms.RandomRotation(10),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

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
        
    def get_transformed_image(self,img):
        return (transform(transforms.ToPILImage(mode='RGB')(img))).unsqueeze(dim=0)

    def get_transformed_images(self,images):

        for idx,img in enumerate(images):
            this=self.get_transformed_image(img)
            if idx==0:
                result=this
            else:
                result=torch.cat((result,this),dim=0)
        return result
    def train(self, net, idea_img,idea_label,delta=None):
        #model = net
        model=dual_model(self.args,net,net)
        net.l2_norm = True
        fixed_model = copy.deepcopy(net)
        # train and update
        idea_img=idea_img.cpu()
        optimizer = optim.SGD(model.parameters(), lr=self.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                
                this_idea_img=self.get_transformed_images(idea_img).to(self.device)
                images=torch.cat((images,this_idea_img),dim=0)
                labels=torch.cat((labels,idea_label),dim=0)
                net.zero_grad()
                logits,activation_loss = model(images,online_target=True)
                log_probs = F.log_softmax(logits/self.args.temp, dim=1)
                loss = self.loss_func(log_probs, labels)+self.alpha*activation_loss

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
