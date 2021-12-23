def build_global_update_module(args):
    if args.global_method == 'base_avg':
        from global_update_method.base_aggregation import GlobalUpdate
    elif args.global_method == 'SlowMo':
        from global_update_method.delta_aggregation_slowmo import GlobalUpdate
    elif args.global_method == 'global_adam':
        from global_update_method.adam_aggregation import GlobalUpdate
    elif args.global_method == 'global_delta':
        from global_update_method.delta_aggregation import GlobalUpdate
    elif args.global_method == 'FedDyn':
        from global_update_method.delta_aggregation_fedDyn import GlobalUpdate
    elif args.global_method == 'FedAGM':
        from global_update_method.delta_aggregation_AGM import GlobalUpdate
    else:
        assert False
        
    return GlobalUpdate