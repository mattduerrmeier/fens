import logging
import os

import pandas as pd
import torch
from aggs import evaluate_all_aggregations
from autoencoder.sampling import convert_tensor_to_dataset, sample_proxy_dataset_tensor
from autoencoder.model import Autoencoder, MseKldLoss
from autoencoder.visualization import visualize_from_dataset, visualize_latent
from mnist_dataset import MNISTDataset
from params.visualization import get_visualization_parameters
from train import evaluate, evaluate_downstream_task, train_and_evaluate
from utils import determine_label_distribution, load_dataset, prepare_client_datasets

import wandb
from args import parse_arguments


def get_parameters(dataset):
    model_config = {}
    if dataset == "FedHeartDisease":
        from flamby.datasets.fed_heart_disease import (
            BATCH_SIZE,
            LR,
            NUM_CLIENTS,
            NUM_EPOCHS_POOLED,
        )
        from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset
        from flamby.datasets.fed_heart_disease import Optimizer

        collate_fn = None
        from loss import BaselineLoss_FHD as BaselineLoss
        from metric import metric_FHD as metric

        from models import Baseline_FHD as Baseline

        BATCH_SIZE = 32
        NUM_CLASSES = 2
        from models import SmallNN_FHD as SmallNN

        model_config = {
            "wide_hidden_dimensions": 11,
            "narrow_hidden_dimensions": 8,
            "latent_dimensions": 3,
        }
    elif dataset == "FedCamelyon16":
        from flamby.datasets.fed_camelyon16 import (
            BATCH_SIZE,
            LR,
            NUM_CLIENTS,
            NUM_EPOCHS_POOLED,
            Baseline,
            BaselineLoss,
        )
        from flamby.datasets.fed_camelyon16 import FedCamelyon16 as FedDataset
        from flamby.datasets.fed_camelyon16 import Optimizer, collate_fn, metric

        NUM_CLASSES = 2
        from models import SmallNN_FCAM as SmallNN
    elif dataset == "FedISIC2019":
        from flamby.datasets.fed_isic2019 import (
            BATCH_SIZE,
            LR,
            NUM_CLIENTS,
            NUM_EPOCHS_POOLED,
            Baseline,
            BaselineLoss,
        )
        from fed_isic_dataset import FedIsicCustom as FedDataset
        from flamby.datasets.fed_isic2019 import Optimizer

        collate_fn = None
        from metric import metric_FISIC as metric

        NUM_CLASSES = 8
        model_config = {
            "wide_hidden_dimensions": 512,
            "narrow_hidden_dimensions": 256,
            "latent_dimensions": 2,
        }
        from models import SmallNN_FISIC as SmallNN
    elif dataset == "MNIST":
        BATCH_SIZE = 128
        LR = 1e-4
        NUM_EPOCHS_POOLED = 5
        NUM_CLIENTS = 2
        Optimizer = torch.optim.Adam
        collate_fn = None
        Baseline = None
        BaselineLoss = None
        metric = None
        FedDataset = MNISTDataset

        NUM_CLASSES = 10
        model_config = {
            "wide_hidden_dimensions": 512,
            "narrow_hidden_dimensions": 256,
            "latent_dimensions": 2,
        }
        from models import SmallNN_MNIST as SmallNN
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
        "nn_model": lambda: SmallNN(NUM_CLIENTS),
        "model_config": model_config,
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
        force=True,
    )

    params = get_parameters(args.dataset)
    visualization_parameters = get_visualization_parameters(args)

    if args.epochs != -1:
        params["num_epochs"] = args.epochs

    # load dataset
    num_clients = params["num_clients"]
    dataset_class = params["fed_dataset"]
    num_classes = params["num_classes"]

    dataset = load_dataset(dataset_class)
    label_distribution: list[int] = determine_label_distribution(dataset)
    num_labels = len(label_distribution)

    logging.info(
        f"Training on dataset with overall label distribution of: {label_distribution}"
    )

    client_datasets, test_dataset = prepare_client_datasets(
        dataset=dataset,
        train_test_split=0.8,
        num_clients=num_clients,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=params["collate_fn"],
    )

    trained_models: list[Autoencoder] = []
    proxy_datasets: list[torch.utils.data.Dataset] = []
    labels_dists: list[list[int]] = []

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    ### 1. train each client's VAE
    for client_idx, client_dataset in zip(range(num_clients), client_datasets):
        print(f"Training VAE {client_idx + 1}/{params['num_clients']}")
        id_str = f"client_{client_idx}"
        logging.info(f"[Training client {client_idx}]")

        proxy_fraction: float = args.proxy_frac

        train_dataset, proxy_dataset = torch.utils.data.random_split(
            client_dataset, (1 - proxy_fraction, proxy_fraction)
        )

        client_label_distribution = determine_label_distribution(train_dataset)
        logging.info(
            f"VAE trains on {len(train_dataset)} records with label distribution {client_label_distribution}"
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            collate_fn=params["collate_fn"],
        )
        proxy_datasets.append(proxy_dataset)
        labels_dists.append(client_label_distribution)

        torch.manual_seed(args.seed + client_idx)

        # encoder is trained on the data and labels
        D_in = len(train_dataset[0][0]) + len(train_dataset[0][1])
        model = Autoencoder(D_in, num_classes, **params["model_config"])

        if args.use_trained_models:
            logging.info(f"Loading trained model {client_idx}")
            model.load_state_dict(
                torch.load(
                    os.path.join(args.trained_models_path, f"{client_idx}_final.pth")
                )
            )
        else:
            loss_fn = MseKldLoss(num_classes, target_coeff=3)
            optimizer = params["optimizer"](model.parameters(), lr=params["lr"])

            # Constant learning rate scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=1.0
            )
            model.to(device)

            best_loss, best_model = train_and_evaluate(
                id_str,
                model,
                loss_fn,
                optimizer,
                scheduler,
                train_dataloader,
                test_dataloader,
                params["num_epochs"],
                device,
            )
            logging.info(f"==> Best Loss for VAE {client_idx + 1}: {best_loss:.4f}")
            model = best_model

            _latents, model_proxy_dataset_tensor = sample_proxy_dataset_tensor(
                [model], 10_000, device
            )
            model_proxy_dataset = convert_tensor_to_dataset(
                model_proxy_dataset_tensor, num_labels
            )

            logging.info(
                f"Generated {len(model_proxy_dataset)} samples from local model model "
                "for training of downstream task "
                f"with label distribution {determine_label_distribution(model_proxy_dataset)} "
            )

            if visualization_parameters.supports_visualization:
                logging.info("Visualizing samples from client model")
                visualize_from_dataset(
                    visualization_parameters.results_path
                    / f"client-{client_idx}-samples.png",
                    model_proxy_dataset,
                )

                logging.info("Visualizing latent space of student model")
                visualize_latent(
                    visualization_parameters.results_path
                    / f"client-{client_idx}-latent.png",
                    model,
                    device,
                )
            else:
                logging.info(
                    "Skip visualization of client model as dataset is not visual"
                )

            print(f"Evaluating VAE {client_idx + 1} on downstream task")
            evaluate_downstream_task(
                id_str,
                torch.utils.data.DataLoader(
                    model_proxy_dataset, batch_size=test_dataloader.batch_size
                ),
                test_dataloader,
                num_classes,
                device,
            )

            if args.save_model:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.result_dir, f"{client_idx}_final.pth"),
                )

        # Add trained model to the list
        trained_models.append(model)

    # evaluation of the VAE
    if not args.use_trained_models:
        client_results = {}
        for client_idx in range(params["num_clients"]):
            print(f"Evaluating VAE {client_idx + 1}/{params['num_clients']}")
            id_str = f"client_{client_idx}"
            logging.info(f"[Evaluating client {client_idx}]")

            loss_fn = MseKldLoss(num_classes, target_coeff=3)
            test_loss, _, _ = evaluate(
                trained_models[client_idx],
                loss_fn,
                test_dataloader,
                device,
            )

            logging.info(f"Loss: {test_loss:.4f}")
            wandb.run.summary[f"{id_str}/best_loss"] = test_loss
            client_results[id_str] = test_loss

    proxy_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(proxy_datasets),
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=params["collate_fn"],
    )

    mse_metric = torch.nn.MSELoss()

    trainable_agg_params = {
        "dataset": args.dataset,
        "lm_lr": args.lm_lr,
        "lm_epochs": args.lm_epochs,
        "nn_lr": args.nn_lr,
        "nn_epochs": args.nn_epochs,
        "distillation_lr": args.distillation_lr,
        "distillation_epochs": args.distillation_epochs,
        "nn_model": params["nn_model"],
        "criterion": mse_metric,
        "model_config": params["model_config"],
    }

    # also tests the downstream tasks with the different aggregation schemes
    agg_results = evaluate_all_aggregations(
        proxy_dataloader,
        test_dataloader,
        trained_models,
        labels_dists,
        num_labels,
        mse_metric,
        device,
        trainable_agg_params,
        visualization_parameters=visualization_parameters,
    )

    # save all results
    # save client results if not loaded from trained models
    all_results = []
    if not args.use_trained_models:
        for client_id, acc in client_results.items():
            all_results.append((client_id, args.seed, acc))

        client_df = pd.DataFrame(
            all_results, columns=["client_id", "seed", "test_loss"]
        )
        client_df.to_csv(
            os.path.join(args.result_dir, "client_results.csv"), index=False
        )

    # save aggregation results
    all_results = []
    for agg, acc in agg_results.items():
        all_results.append(
            (
                agg,
                args.seed,
                acc["mse_loss"],
                acc["downstream_train_accuracy"],
                acc["downstream_test_accuracy"],
            )
        )

    agg_df = pd.DataFrame(
        all_results,
        columns=[
            "agg",
            "seed",
            "mse_loss",
            "downstream_train_accuracy",
            "downstream_test_accuracy",
        ],
    )
    agg_df.to_csv(os.path.join(args.result_dir, "agg_results.csv"), index=False)

    logging.info("Saved successfully")


if __name__ == "__main__":
    args = parse_arguments()

    gpu_idx = args.gpu_idx
    # device = torch.device("cpu")
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")

    # extract name from args.result_dir
    run_name = os.path.basename(args.result_dir)

    # initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=args,
        mode="disabled" if args.disable_wandb else "online",
    )

    run(args, device)

    wandb.finish()
