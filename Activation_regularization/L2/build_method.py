def build_local_update_module(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.method == 'Fedavg':
        from local_update_method.base import LocalUpdate
    if args.method == 'l2_act_reg':
        from local_update_method.l2_activation import LocalUpdate
    else:
        assert False

    return LocalUpdate