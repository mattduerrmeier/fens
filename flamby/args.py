import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for FLAMBY')
    
    parser.add_argument('--dataset', type=str, choices=['FedHeartDisease', 'FedCamelyon16', 'FedISIC2019'], \
                        default='FedHeartDisease', help='Dataset')
    parser.add_argument('--epochs', type=int, default=-1, help='Number of epochs; -1 defaults to flamby')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU device')
    parser.add_argument('--result_dir', type=str, default='results', help='Result directory')
    parser.add_argument('--proxy_frac', type=float, default=0.1, help='Proxy fraction') 
    parser.add_argument('--save_model', action='store_true', help='Save model')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO'], 
                        default='INFO', help='Log level')
    parser.add_argument('--test_every', type=int, default=2, help='Test every n epochs')
    parser.add_argument('--lm_epochs', type=int, default=300, help='Linear mapping agg epochs')
    parser.add_argument('--lm_lr', type=float, default=1e-5, help='Linear mapping agg learning rate')
    parser.add_argument('--nn_epochs', type=int, default=300, help='Neural network agg epochs')
    parser.add_argument('--nn_lr', type=float, default=1e-5, help='Neural network agg learning rate')
    parser.add_argument('--use_trained_models', action='store_true', help='Use trained models')
    parser.add_argument('--trained_models_path', type=str, default=None, help='Trained models path')
    parser.add_argument('--wandb_project', type=str, default='fens', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='fens', help='Wandb entity name')

    args = parser.parse_args()

    if args.use_trained_models and args.trained_models_path is None:
        parser.error("--use_trained_models requires --trained_models_path")
    
    return args