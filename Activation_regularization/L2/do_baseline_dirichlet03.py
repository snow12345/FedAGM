import os
import subprocess
import shlex
#iid_common=['python3','main.py']
iid_common=['nohup','python3','federated_train.py', '--config', 'configs/cifar_Fedavg.yaml','--data', './data','--mode','dirichlet','--dirichlet_alpha','0.3','--batch_size','50',"--centralized_epochs",'0',"--global_epochs",'1000','--local_epochs','5','--epsilon','0.1','--momentum','0','--lr','0.1','--learning_rate_decay','0.998','--weight_decay','1e-3','--seed','0','--set','CIFAR10','--arch','ResNet18','--workers','0','--cuda_visible_device','2']
###################################################3
this_argv=iid_common
this_alpha=0
this_argv.append('--alpha')
this_argv.append(str(this_alpha))
this_argv.append('--additional_experiment_name')
this_argv.append('FedAvg balanced')
print(this_argv)
subprocess.Popen(this_argv) 
#####################################################
