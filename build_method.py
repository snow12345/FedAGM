def build_local_update_module(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.method == 'Fedavg':
        from local_update_method.base import LocalUpdate
    elif args.method == 'FedProx':
        from local_update_method.weight_l2 import LocalUpdate
    elif args.method == 'FedCM':
        from local_update_method.fedCM import LocalUpdate
    elif args.method == 'FedDyn':
        from local_update_method.fedDyn import LocalUpdate
    elif args.method == 'FedAGM':
        from local_update_method.fedAGM import LocalUpdate
    else:
        assert False

    return LocalUpdate