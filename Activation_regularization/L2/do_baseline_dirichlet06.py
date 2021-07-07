import os
import subprocess
import shlex

#iid_common=['python3','main.py']
iid_common=['python3','federated_train.py', '--config', 'configs/cifar_Fedavg.yaml','--data', './data','--mode','dirichlet','--dirichlet_alpha','0.6','--batch_size','50',"--centralized_epochs",'0',"--global_epochs",'1000','--local_epochs','5','--epsilon','0.1','--momentum','0','--lr','0.1','--learning_rate_decay','1','--weight_decay','1e-3','--seed','0','--set','CIFAR10','--arch','CNN','--workers','0','--cuda_visible_device','1']
###################################################3
this_argv=iid_common
this_alpha=0
this_argv.append('--alpha')
this_argv.append(str(this_alpha))
this_argv.append('--additional_experiment_name')
this_argv.append('alpha'+str(this_alpha)+'_zeromomentum')
print(this_argv)
subprocess.Popen(this_argv) 
#####################################################
