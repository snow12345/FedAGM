def build_local_update_module(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.method == 'Fedavg':
        from local_update_method.base import LocalUpdate
    elif args.method == 'Fedavg_sign':
        from local_update_method.base_sign import LocalUpdate
    elif args.method == 'Fedavg_temp':
        from local_update_method.base_temp import LocalUpdate
    elif args.method == 'Fedavg_learn_wrong':
        from local_update_method.base_lw import LocalUpdate
    elif args.method == 'Fedavg_mask':
        from local_update_method.base_masking import LocalUpdate
    elif args.method == 'Fedavg_mask_sign':
        from local_update_method.base_masking_sign import LocalUpdate
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
    elif args.method == 'simsiam':
        from local_update_method.base_simsiam import LocalUpdate
    elif args.method == 'FedCSAM':
        from local_update_method.FedCSAM import LocalUpdate
    elif args.method == 'FedPReg_and_PodNet':
        from local_update_method.proxy_reg_And_podnet import LocalUpdate    
  
    elif args.method == 'FedPReg_and_PodNet_CM':
        from local_update_method.proxy_reg_And_podnet_CM import LocalUpdate  
    elif args.method == 'proto':
        from local_update_method.proto import LocalUpdate    
    elif args.method == 'proto_scratch':
        from local_update_method.proto_scratch import LocalUpdate            
    elif args.method == 'FedPReg_and_PodNet_proto_soft':
        from local_update_method.proxy_reg_And_podnet_proto_soft import LocalUpdate            
        
    elif args.method == 'FedPReg_and_PodNet_proto':
        from local_update_method.proxy_reg_And_podnet_proto import LocalUpdate    
    elif args.method == 'FedPReg_and_PodNet_proto_separate':
        from local_update_method.proxy_reg_And_podnet_proto_separate import LocalUpdate    
    elif args.method == 'FedPReg_and_PodNet_proto_noce':
        from local_update_method.proxy_reg_And_podnet_proto_noce import LocalUpdate    
    elif args.method == 'FedPReg_and_PodNet_proto_DACM':
        from local_update_method.proxy_reg_And_podnet_DACM import LocalUpdate   
    else:
        assert False

    return LocalUpdate