def build_global_update_module(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.global_method == 'global_avg':
        from global_update_method.base_aggregation import GlobalUpdate
    elif args.global_method == 'global_avg_momentum':
        from global_update_method.delta_aggregation_momentum import GlobalUpdate
    elif args.global_method == 'global_avg_momentum2':
        from global_update_method.delta_aggregation_momentum2 import GlobalUpdate
    elif args.global_method == 'global_adam':
        from global_update_method.adam_aggregation import GlobalUpdate
    elif args.global_method == 'global_adam_proto':
        from global_update_method.adam_aggregation_proto import GlobalUpdate
    elif args.global_method == 'global_delta':
        from global_update_method.delta_aggregation import GlobalUpdate
    elif args.global_method == 'global_delta_rand':
        from global_update_method.delta_aggregation_rand import GlobalUpdate
    elif args.global_method == 'global_delta_topk':
        from global_update_method.topk_delta_aggregation import GlobalUpdate
    elif args.global_method == 'global_delta_ema':
        from global_update_method.ema_delta_aggregation import GlobalUpdate
    elif args.global_method == 'byol':
        from global_update_method.base_aggregation_byol import GlobalUpdate
    elif args.global_method == 'global_proto':
        from global_update_method.base_aggregation_proto import GlobalUpdate
    elif args.global_method == 'global_proto_generator':
        from global_update_method.base_aggregation_proto_generator import GlobalUpdate
    elif args.global_method == 'global_proto_separate':
        from global_update_method.base_aggregation_proto_separate import GlobalUpdate
    elif args.global_method == 'global_proto2':
        from global_update_method.base_aggregation_proto2 import GlobalUpdate
    elif args.global_method == 'global_proto3':
        from global_update_method.base_aggregation_proto3 import GlobalUpdate
    elif args.global_method == 'global_proto_scratch':
        from global_update_method.base_aggregation_proto_scratch import GlobalUpdate        
    elif args.global_method == 'global_proto_soft':
        from global_update_method.base_aggregation_proto_soft import GlobalUpdate        
    elif args.global_method == 'global_protoserver':
        from global_update_method.base_aggregation_protoserver import GlobalUpdate
    elif args.global_method == 'FedDyn':
        from global_update_method.delta_aggregation_fedDyn import GlobalUpdate

    elif args.global_method == 'FedNAG':
        from global_update_method.delta_aggregation_NAG import GlobalUpdate
    elif args.global_method == 'Aggmo':
        from global_update_method.delta_aggregation_Aggmo import GlobalUpdate        
    elif args.global_method == 'FedNAG_ctr_oversh':
        from global_update_method.delta_aggregation_NAG_ctr_oversh import GlobalUpdate  
    elif args.global_method == 'FedNAG_serverlr':
        from global_update_method.delta_aggregation_NAG_serverlr import GlobalUpdate        
        
    elif args.global_method == 'FedNAG_negative_similarity':
        from global_update_method.delta_aggregation_NAG_negative_similarity import GlobalUpdate
    else:
        assert False



        
    return GlobalUpdate
    '''
    elif args.method == 'FedProx':
        from local_update_method.weight_l2 import LocalUpdate
    '''