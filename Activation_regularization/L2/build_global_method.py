def build_global_update_module(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.global_method == 'global_avg':
        from global_update_method.base_aggregation import GlobalUpdate
    elif args.global_method == 'global_adam':
        from global_update_method.adam_aggregation import GlobalUpdate
    elif args.global_method == 'global_delta':
        from global_update_method.delta_aggregation import GlobalUpdate
    elif args.global_method == 'global_delta_ema':
        from global_update_method.ema_delta_aggregation import GlobalUpdate
    elif args.global_method == 'byol':
        from global_update_method.base_aggregation_byol import GlobalUpdate

    else:
        assert False

    return GlobalUpdate
    '''
    elif args.method == 'FedProx':
        from local_update_method.weight_l2 import LocalUpdate
    '''