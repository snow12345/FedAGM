import os
import subprocess
import shlex

iid_common=['nohup','python3','federated_train.py', '--config', 'configs/cifar_Fedavg.yaml','--data', './data','--mode','dirichlet','--dirichlet_alpha','0.3','--batch_size','50',"--centralized_epochs",'0',"--global_epochs",'1000','--local_epochs','5','--epsilon','0.1','--momentum','0','--lr','0.1','--learning_rate_decay','1','--weight_decay','1e-3','--seed','0','--set','CIFAR10','--arch','CNN','--workers','8','--cuda_visible_device','2']
###################################################
for i in range(7):
	print(i)
	this_argv=iid_common
	this_alpha=10**(-i)
	this_argv.append('--alpha')
	this_argv.append(str(this_alpha))
	this_argv.append('--additional_experiment_name')
	this_argv.append('alpha'+str(this_alpha)+'_zeromomentum')

	subprocess.Popen(this_argv)

#####################################################


this_argv=iid_common
this_alpha=0.001
this_argv.append('--alpha')
this_argv.append(str(this_alpha))
this_argv.append('--alpha_mul_epoch')
this_argv.append('--additional_experiment_name')
this_argv.append('alpha'+str(this_alpha)+'alpha_mul_epoch'+'_zeromomentum')


subprocess.Popen(this_argv)
#####################################################


this_argv=iid_common
this_alpha=1
this_argv.append('--alpha')
this_argv.append(str(this_alpha))
this_argv.append('--alpha_divide_epoch')
this_argv.append('--additional_experiment_name')
this_argv.append('alpha'+str(this_alpha)+'alpha_divide_epoch'+'_zeromomentum')


subprocess.Popen(this_argv)


