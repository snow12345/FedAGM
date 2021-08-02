def build_local_update_module(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.method == 'Fedavg':
        from local_update_method.base import LocalUpdate
    elif args.method == 'l2_act_reg':
        from local_update_method.l2_activation import LocalUpdate
    elif args.method == 'FedProx':
        from local_update_method.weight_l2 import LocalUpdate
    elif args.method == 'PodNet':
        from local_update_method.podnet import LocalUpdate
    elif args.method == 'FedCM':
        from local_update_method.fedCM import LocalUpdate
    elif args.method == 'FedCADAM':
        from local_update_method.fedCADAM import LocalUpdate
    elif args.method == 'FedPReg':
        from local_update_method.proxy_reg import LocalUpdate

    elif args.method == 'byol':
        from local_update_method.base_byol import LocalUpdate
    elif args.method == 'FedCSAM':
        from local_update_method.FedCSAM import LocalUpdate
    elif args.method == 'FedPReg_and_PodNet':
        from local_update_method.proxy_reg_And_podnet import LocalUpdate    
    else:
        assert False

    return LocalUpdate