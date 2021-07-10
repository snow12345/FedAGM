import argparse
import sys
import yaml
#from configs import parser as _parser


USABLE_TYPES = set([float, int])


def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def produce_override_string(args, override_args):
    lines = []
    for v in override_args:
        if v != "multigpu":
            v_arg = getattr(args, v)
            if type(v_arg) in USABLE_TYPES:
                lines.append(v + ": " + str(v_arg))
            else:
                lines.append(v + ": " + f'"{str(v_arg)}"')
        else:
            lines.append("multigpu: " + str(args.multigpu))

    return "\n# ===== Overrided ===== #\n" + "\n".join(lines)


args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--mode", help="data setting to use", default="iid"
    )
    parser.add_argument(
        "--project", help="data setting to use", default="federated_learning"
    )
    parser.add_argument(
        "--method", help="method to use", default="l2_act_reg"
    )
    parser.add_argument(
        "--global_method", help="aggregation method to use", default="global_avg"
    )
    parser.add_argument(
        "--additional_experiment_name", help="specific information for logging", default=""
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--cuda_visible_device",
        default=0,
        type=float,
        metavar="N",
        help="which GPU to use",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--create_client_dataset",
        action="store_true",
        help="Whether or not to make new client dataset",
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )

    ## Training Config
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="CNN", help="model architecture"
    )
    parser.add_argument(
        "--global_epochs",
        default=1000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--centralized_epochs",
        default=500,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--local_epochs",
        default=5,
        type=int,
        metavar="N",
        help="number of local epochs to run on each global epoch",
    )
    parser.add_argument(
        "--num_of_clients",
        default=100,
        type=int,
        metavar="N",
        help="number of total clients",
    )
    parser.add_argument(
        "--participation_rate",
        default=0.1,
        type=float,
        metavar="N",
        help="number of participation rate at each global epoch",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=50,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )

    parser.add_argument(
        "--momentum", default=0.0, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0.0)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--ld",
        "--learning_rate_decay",
        default=1.0,
        type=float,
        metavar="LD",
        help="learning rate decay (default: 1.0)",
        dest="learning_rate_decay",
    )
    parser.add_argument(
        "--gr_clipping_max_norm",
        default=10,
        type=int,
        metavar="N",
        help="gradient clippling max norm",
    )

    ## Dataset
    parser.add_argument("--set", help="name of dataset", type=str, default="CIFAR10")
    parser.add_argument(
        "--dirichlet_alpha",
        default=-0.6,
        type=float,
        metavar="N",
        help="federated dirichlet alpha",
    )

    parser.add_argument(
        "--alpha_mul_epoch",
        action="store_true",
        help="Whether or not to multiplicate alpha with epoch",
    )
    parser.add_argument(
        "--alpha_divide_epoch",
        action="store_true",
        help="Whether or not to divide alpha by epoch",
    )
    parser.add_argument(
        "--data", help="path to dataset base directory", default="./data"
    )
    parser.add_argument(
        "--client_data", help="path to client dataset base directory", default="./client_data"
    )
    parser.add_argument("--num-classes", default=10, type=int)



    ## Eval

    parser.add_argument(
        "-p",
        "--print-freq",
        default=1,
        type=int,
 
        help="print frequency (default: 1)",
    )


    ### Method Hyperparameters

    ## l2-act-reg
    parser.add_argument(
        "--alpha",
        default=0.0,
        type=float,

        help="federated activation regularization hyperparameter alpha",
    )
    ## pod
    parser.add_argument(
        "--pod_normalize",
        action="store_false",
        help="Whether or not to normalize activation in pod",
    )
    parser.add_argument(
        "--collapse_channels",
        help="Whether or not to normalize activation in pod",
        default='spatial'
        ####"channels" "width" "height" "gap" "spatial"
    )
    
    

    ## l2-weight
    parser.add_argument(
        "--mu",
        default=0.0,
        type=float,
        metavar="N",
        help="federated weight regularization hyperparameter mu",
    )
    
    ## FedAdam
    parser.add_argument(
        "--beta_1",
        
        default=0.9,
        type=float,
        metavar="N",
        help="FedAdam beta_1",
        
    )
    parser.add_argument(
        "--beta_2",
        
        default=0.99,
        type=float,
        metavar="N",
        help="FedAdam beta_2",
        
    )
    parser.add_argument(
        "--eta",
        
        default=1e-2,
        type=float,
        metavar="N",
        help="FedAdam server learning rate",
        
    )
    parser.add_argument(
        "--server_learning_rate_decay",
        
        default=1,
        type=float,
        metavar="N",
        help="FedAdam server learning rate decay",
        
    )    
    parser.add_argument(
        "--tau",
        
        default=1e-3,
        type=float,
        metavar="N",
        help="FedAdam beta_2",
        
    )
    ## ETC

    parser.add_argument(
        "--epsilon",
        default=0.0,
        type=float,
        metavar="N",
        help="Used in LSC Loss",
    )
    '''

    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    '''

    '''
    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--low-data", default=1, help="Amount of data to use", type=float
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )


    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )
    parser.add_argument(
        "--trainer", type=str, default="default", help="cs, ss, or standard training"
    )
    '''

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
