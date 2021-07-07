import os
import subprocess
import shlex

iid_common=['nohup','python3','main.py', '--config', 'configs/cifar_Fedavg.yaml','--data', './data','--mode','iid','--dirichlet_alpha','0.6','--batch_size','50',"--centralized_epochs",'0',"--global_epochs",'1000','--local_epochs','5','--epsilon','0.1','--momentum','0','--lr','0.1','--learning_rate_decay','0.992','--weight_decay','1e-3','--seed','0','--set','CIFAR10','--arch','CNN','--workers','8','--cuda_visible_device','3']
###################################################
for i in range(5):
	print(i)
	this_argv=iid_common
	this_alpha=10**(-i)
	this_argv.append('--alpha')
	this_argv.append(str(this_alpha))
	this_argv.append('--additional_experiment_name')
	this_argv.append('alpha'+str(this_alpha)+'lrdecay0.992ldbugfixed')

	subprocess.Popen(this_argv)

#####################################################

this_argv=iid_common
this_alpha=0.0001
this_argv.append('--alpha')
this_argv.append(str(this_alpha))
this_argv.append('--alpha_mul_epoch')
this_argv.append('--additional_experiment_name')
this_argv.append('alpha'+str(this_alpha)+'alpha_mul_epoch'+'lrdecay0.992ldbugfixed')


subprocess.Popen(this_argv)
#####################################################


this_argv=iid_common
this_alpha=0.1
this_argv.append('--alpha')
this_argv.append(str(this_alpha))
this_argv.append('--alpha_divide_epoch')
this_argv.append('--additional_experiment_name')
this_argv.append('alpha'+str(this_alpha)+'alpha_divide_epoch'+'lrdecay0.992ldbugfixed')


subprocess.Popen(this_argv)


