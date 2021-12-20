# Federated Learning with Acceleration of Global Momentum

CIFAR-100, 100 clients, Dirichlet (0.3) split, 5% participation
~~~
python federated_train.py --cuda_visible_device=1 --method=FedAGM --global_method=FedAGM --config=configs/cifar_actl2.yaml --arch=ResNet18 --weight_decay=1e-3 --gr_clipping_max_norm=10 --alpha=1 --mu=0.001 --gamma=0.9 --momentum=0.0 --tau 1.0 --lr=1e-1 --mode=dirichlet --dirichlet_alpha=0.3  --participation_rate=0.05 --learning_rate_decay 0.995 --set CIFAR100
~~~
