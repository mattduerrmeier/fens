from args import parse_arguments
import torch
import os
import logging
import wandb
import pandas as pd

from utils import load_trainset_combined, generate_logits_combined
from train import train_and_evaluate, evaluate
from aggs import evaluate_all_aggregations

def get_parameters(dataset):
    if dataset == "FedHeartDisease":
        from flamby.datasets.fed_heart_disease import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            NUM_CLIENTS,
            Optimizer,
            FedHeartDisease as FedDataset
        )
        collate_fn = None
        from metric import metric_FHD as metric
        from models import Baseline_FHD as Baseline
        from loss import BaselineLoss_FHD as BaselineLoss
        NUM_CLASSES = 2
        require_argmax = False
        from models import SmallNN_FHD as SmallNN
    elif dataset == "FedCamelyon16":
        from flamby.datasets.fed_camelyon16 import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            NUM_CLIENTS,
            Optimizer,
            collate_fn,
            Baseline,
            BaselineLoss,
            metric,
            FedCamelyon16 as FedDataset
        )
        NUM_CLASSES = 2
        require_argmax = False
        from models import SmallNN_FCAM as SmallNN
    elif dataset == "FedISIC2019":
        from flamby.datasets.fed_isic2019 import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            NUM_CLIENTS,
            Optimizer,
            Baseline,
            BaselineLoss,
            FedIsic2019 as FedDataset
        )
        collate_fn = None
        from metric import metric_FISIC as metric
        NUM_CLASSES = 8
        require_argmax = True
        from models import SmallNN_FISIC as SmallNN
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    params = {
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "num_epochs": NUM_EPOCHS_POOLED,
        "num_clients": NUM_CLIENTS,
        "optimizer": Optimizer,
        "num_classes": NUM_CLASSES,
        "fed_dataset": FedDataset,
        "collate_fn": collate_fn,
        "baseline": Baseline,
        "baseline_loss": BaselineLoss,
        "metric": metric,
        "require_argmax": require_argmax,
        "nn_model": SmallNN,
    }
    
    return params

def run(args, device):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)    

    log_level = logging.DEBUG if args.log_level == "DEBUG" else logging.INFO

    log_file = os.path.join(args.result_dir, "log.txt")
    logging.basicConfig(
        filename=log_file, 
        level=log_level,
        format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s",
        force=True
    )

    p = get_parameters(args.dataset)
    FedDataset = p["fed_dataset"]

    if args.epochs != -1:
        p["num_epochs"] = args.epochs

    # create test dataset
    testset = FedDataset(train=False, pooled=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=p["batch_size"], 
        shuffle=False, num_workers=0, collate_fn=p["collate_fn"])
    
    trained_models = []
    proxy_datasets = []
    label_dists = []

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        
    for i in range(p["num_clients"]):
        id_str = f'client_{i}'
        logging.info(f'[Training client {i}]')
        
        train_dataset, proxy_dataset, label_dist = \
            load_trainset_combined(args.dataset, i, FedDataset, p["num_classes"], 
                                   proxy_frac=args.proxy_frac, seed=args.seed)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=p["batch_size"], shuffle=True, 
            num_workers=1, collate_fn=p["collate_fn"])
        proxy_datasets.extend(proxy_dataset)
        label_dists.append(label_dist)

        torch.manual_seed(args.seed + i)
        model = p["baseline"]()
        
        if args.use_trained_models:
            logging.info(f'Loading trained model {i}')
            model.load_state_dict(torch.load(os.path.join(args.trained_models_path, f'{i}_final.pth')))
        else:
            lossfunc = p["baseline_loss"]()
            optimizer = p["optimizer"](model.parameters(), lr=p["lr"])

            # Constant learning rate scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
            model.to(device)

            best_acc, best_model = train_and_evaluate(id_str, model, lossfunc, optimizer, scheduler, 
                                train_dataloader, test_dataloader, p["num_epochs"], device, 
                                test_every=args.test_every, metric=p["metric"],
                                require_argmax=p["require_argmax"])
            logging.info(f'Best accuracy: {best_acc}')
            
            model = best_model 
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.result_dir, f'{i}_final.pth'))

        # Add trained model to the list
        trained_models.append(model)    

    # Evaluation
    if not args.use_trained_models:
        client_results = {}
        for i in range(p["num_clients"]):
            id_str = f'client_{i}'
            logging.info(f'[Evaluating client {i}]')
            test_loss, test_acc = evaluate(trained_models[i], p["baseline_loss"](), 
                                        test_dataloader, device, p["metric"], 
                                        require_argmax=p["require_argmax"])
            logging.info(f'Loss: {test_loss:.4f} Accuracy: {test_acc:.4f}')
            wandb.run.summary[f'{id_str}/best_acc'] = test_acc
            wandb.run.summary[f'{id_str}/best_loss'] = test_loss
            client_results[id_str] = test_acc

        logits = generate_logits_combined(args.dataset, trained_models, 
        p["fed_dataset"], p["num_classes"], args.proxy_frac, device,
        test_dataloader, p["num_clients"], p["batch_size"], args.seed, 
        p["collate_fn"])

    proxy_dataloader = torch.utils.data.DataLoader(proxy_datasets, batch_size=p["batch_size"], 
        shuffle=True, num_workers=0, collate_fn=p["collate_fn"])
    
    trainable_agg_params = {
        'lm_lr': args.lm_lr,
        'lm_epochs': args.lm_epochs,
        'nn_lr': args.nn_lr,
        'nn_epochs': args.nn_epochs,
        'nn_model': p["nn_model"],
        'criterion': p["baseline_loss"](),
    }

    agg_results = evaluate_all_aggregations(
        proxy_dataloader, 
        test_dataloader, 
        trained_models,
        label_dists, 
        p["metric"],
        device,
        trainable_agg_params,
        require_argmax=p["require_argmax"],
    )        

    # save all results
    # save client results if not loaded from trained models
    all_results = []
    if not args.use_trained_models:
        for client_id, acc in client_results.items():
            all_results.append((client_id, args.seed, acc))
        client_df = pd.DataFrame(all_results, columns=['client_id', 'seed', 'accuracy'])
        client_df.to_csv(os.path.join(args.result_dir, 'client_results.csv'), index=False)

        # save logits
        logits_file = os.path.join(args.result_dir, 'logits.pth')
        torch.save(logits, logits_file, pickle_protocol=2)
        
    # save aggregation results
    all_results = []
    for agg, acc in agg_results.items():
        all_results.append((agg, args.seed, acc))
    agg_df = pd.DataFrame(all_results, columns=['agg', 'seed', 'accuracy'])
    agg_df.to_csv(os.path.join(args.result_dir, 'agg_results.csv'), index=False)

    logging.info('Saved successfully')

if __name__ == "__main__":
    args = parse_arguments()

    gpu_idx = args.gpu_idx
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")

    # extract name from args.result_dir
    run_name = os.path.basename(args.result_dir)

    # initialize wandb
    wandb.init(
        project=args.wandb_project, 
        entity=args.wandb_entity,
        name=run_name, 
        config=args
    )

    run(args, device)

    wandb.finish()