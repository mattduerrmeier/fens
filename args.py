import argparse

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
    parser.add_argument('--name','-n', 
                        default="default", 
                        type=str, 
                        help='experiment name, used for saving results')
    
    # wandb
    parser.add_argument('--wandb_project',
                        default='fens',
                        type=str,
                        help='wandb project name')
    parser.add_argument('--wandb_entity',
                        default='fens',
                        type=str,
                        help='wandb entity name')

    # dataset
    parser.add_argument('--alpha', 
                        default=0.2, 
                        type=float, 
                        help='control the non-iidness of dataset')
    parser.add_argument('--p', '-p', 
                        action='store_true', 
                        help='whether the dataset is partitioned or not')
    parser.add_argument('--NIID',
                        action='store_true',
                        help='whether the dataset is non-iid or not')
    parser.add_argument('--dataset',
                        default='CIFAR10',
                        type=str,
                        choices=['CIFAR10', 'SVHN', 'CIFAR10Agg', 'SVHNAgg', \
                                 'FedHeartDiseaseAgg', 'FedISIC2019Agg', 'SVHNBasic', \
                                'SVHNBasicAgg', 'CIFAR100', 'CIFAR100Agg', 'FedCamelyon16Agg', \
                                'CIFAR10QAgg', 'CIFAR100QAgg', 'SVHNBasicQAgg', 'CIFAR10QAggAdv', \
                                'CIFAR100QAggAdv', 'SVHNBasicQAggAdv'],
                        help='which dataset to run on')
    parser.add_argument('--val_set',
                        action='store_true',
                        help='create a validation set from the training set')
    parser.add_argument('--val_ratio', 
                        default=0.5, 
                        type=float, 
                        help='fraction of testing data to be used as validation set')
    parser.add_argument('--proxy_set',
                        action='store_true',
                        help='create a proxy set from the training set')
    parser.add_argument('--proxy_ratio',
                        default=0.1,
                        type=float,
                        help='fraction of training data to be used as proxy set')
    parser.add_argument('--include_trainset', # for Agg datasets
                        action='store_true',
                        help='include a fraction of the training set in the proxy set for Agg datasets')
    parser.add_argument('--include_trainset_frac', # for Agg datasets
                        default=0.1,
                        type=float,
                        help='fraction of training set to be included in the proxy set for Agg datasets')
    parser.add_argument('--tr_subset',
                        action='store_true',
                        help='use a subset of the training set')
    parser.add_argument('--tr_subset_frac',
                        default=0.8,
                        type=float,
                        help='size of the training subset')
    parser.add_argument('--apply_augmentation',
                        type=bool,
                        default=True,
                        help='apply data augmentation during training')
    
    # training configuration
    parser.add_argument('--model', 
                        default="ResNet8",
                        choices=['ResNet8', 'VGG', 'SmallNN', 'SmallNN_FHD', 'WeightedNN', \
                                'SmallNN_FCAM', 'FederatedMoE', 'FederatedMoE2', \
                                'LinearAggregator'], 
                        type=str, 
                        help='neural network model')
    parser.add_argument('--d',
                        default=4,
                        type=int,
                        help='parameter of the small nn aggregator network')
    parser.add_argument('--diff_init',
                        action='store_true',
                        help='use different initial model parameters for all clients')
    parser.add_argument('--lr', 
                        default=0.1, 
                        type=float, 
                        help='client learning rate')
    parser.add_argument('--slr', 
                        default=1.0, 
                        type=float, 
                        help='server learning rate')
    parser.add_argument('--updatelr',  
                        action='store_true', 
                        help='Decays learning rate if set')
    parser.add_argument('--gmf', 
                        default=0, 
                        type=float, 
                        help='global (server) momentum factor')
    parser.add_argument('--momentum', 
                        default=0.0, 
                        type=float, 
                        help='local (client) momentum factor')
    parser.add_argument('--bs', 
                        default=512, 
                        type=int, 
                        help='batch size on each worker/client')
    parser.add_argument('--test_bs',
                        default=32,
                        type=int,
                        help='batch size for testing')
    parser.add_argument('--rounds', 
                        default=200, 
                        type=int, 
                        help='total coommunication rounds')
    parser.add_argument('--lowE', 
                        default=2, 
                        type=int, 
                        help='lower bound for number of local epochs/iterations')
    parser.add_argument('--highE', 
                        default=2, 
                        type=int, 
                        help='upper bound for number of local epochs/iterations')
    parser.add_argument('--isEpochs', '-iE', 
                        action='store_true', 
                        help='whether specified [low,high]E are epochs or iterations')
    parser.add_argument('--mu', 
                        default=0, 
                        type=float, 
                        help='mu parameter in fedprox')
    parser.add_argument('--optimizer', 
                        default='fedavg', 
                        type=str, 
                        help='optimizer name')
    parser.add_argument('--use_scheduler',
                        action='store_true',
                        help='use learning rate scheduler during local training')
    parser.add_argument('--weights',
                        type=str,
                        choices=['uniform', 'data_based'],
                        default='data_based',
                        help='weights for aggregating client models')
    parser.add_argument('--stop_early',
                        action='store_true',
                        help='terminate early based on fixed criteria if accuracy is not improving')
    
    # distributed training configuration
    parser.add_argument('--initmethod',
                        default='tcp://h0:22000',
                        type=str,
                        help='init method')
    parser.add_argument('--backend',
                        default="nccl",
                        type=str,
                        help='background name')
    parser.add_argument('--procs_per_machine', 
                        default=4, 
                        type=int, 
                        help='number of processes per machine')
    parser.add_argument('--rank', 
                        default=0, 
                        type=int, 
                        help='the rank of worker')
    parser.add_argument('--size', 
                        default=8, 
                        type=int, 
                        help='number of local workers')
    parser.add_argument('--numclients', 
                        default=8, 
                        type=int, 
                        help='clients per round, always same as size parameter; kept for compatibility')
    parser.add_argument('--totalclients', 
                        default=100, 
                        type=int, 
                        help='total clients to split the dataset across [for CIFAR10, SVHN]')
    parser.add_argument('--gpu', '-g', 
                        action='store_true', # default value is False
                        help='whether to run on gpu')
    
    # other
    parser.add_argument('--print_freq', 
                        default=100, 
                        type=int, 
                        help='print info frequency')
    parser.add_argument('--evalafter', 
                        default=1, 
                        type=int, 
                        help='number of communication rounds to evaluate after')
    parser.add_argument('--max_itr', 
                        default=0, 
                        type=int, 
                        help='threshold on the number of local iterations when running epochs')
    parser.add_argument('--seed', 
                        default=1, 
                        type=int, 
                        help='random seed')
    parser.add_argument('--save', '-s', 
                        action='store_true', 
                        help='whether save the training results')
    parser.add_argument('--save_model', '-sm', 
                        action='store_true', 
                        help='whether to save the model and label distribution after training')
    parser.add_argument('--debug',
                        action='store_true',
                        help='enable debug mode')
    parser.add_argument('--disable_wandb',
                        action='store_true',
                        help='disable logging to wandb')
    
    # paths
    parser.add_argument('--savepath',
                        default='./results/',
                        type=str,
                        help='directory to save exp results')
    parser.add_argument('--logitpath', # for Agg datasets
                        default=None,
                        type=str,
                        help='directory from where to load trained model for aggregator training')
    parser.add_argument('--datapath',
                        default='./data/',
                        type=str,
                        help='directory to load data')
    parser.add_argument('--model_init_path',
                        default=None,
                        type=str,
                        help='path to the initial model for multi-shot FL')
    
    args = parser.parse_args()

    return args