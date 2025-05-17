import torch
from aggregators import common, distillation, neural
from aggregators.average import averaging
from aggregators.linear import linear_mapping
from typing import Union

import wandb


def _determine_ensemble_proxy_dataset(
    models: list[torch.nn.Module],
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    proxy_dataset = []
    # we select the X_hat predicted by each vae, as well as the input X
    with torch.no_grad():
        for elems, labels in loader:
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(dim=1).to(device)

            assert len(labels.shape) == 2

            elems = elems.to(device)
            labels = labels.to(device)

            outputs: list[torch.Tensor] = []
            for model in models:
                model_output, _latent_mean, _latent_variance = model((elems, labels))

                outputs.append(model_output.detach().cpu())

            stacked_outputs = torch.stack(outputs)

            elems = elems.cpu()
            labels = labels.cpu()
            data = torch.concat((elems, labels), dim=1)

            proxy_dataset.append((stacked_outputs, data))

    return proxy_dataset


def evaluate_all_aggregations(
    train_loader,
    test_loader,
    models,
    num_labels,
    metric,
    device,
    trainable_agg_params,
    require_argmax=False,
):
    # elems: (batch_size, input_dim)
    # model(elems): (batch_size, output_dim)
    # hstack(outputs): (batch_size, output_dim * num_models)

    # move models to device
    for model in models:
        model = model.to(device)

    # Create the dataset of predictions for training and testing
    trainset = _determine_ensemble_proxy_dataset(
        models=models, loader=train_loader, device=device
    )

    testset = _determine_ensemble_proxy_dataset(
        models=models, loader=test_loader, device=device
    )

    results = {}

    avg_performance = averaging(
        testset, metric, len(models), num_labels, require_argmax
    )
    results["avg"] = {
        "mse_loss": avg_performance,
        "downstream_train_accuracy": -1,
        "downstream_test_accuracy": -1,
    }

    # TODO: fix weighted averaging
    if False:
        wavg_performance = weighted_averaging(
            testset,
            metric,
            len(models),
            len(num_labels[0]),
            num_labels,
            require_argmax,
        )
        results["wavg"] = wavg_performance

    if False:
        voting_performance = polychotomous_voting(
            testset,
            metric,
            len(models),
            num_labels,
            models,
            device,
            trainset,
            require_argmax,
        )
        results["voting"] = voting_performance

    lm_performance = linear_mapping(
        testset,
        metric,
        len(models),
        num_labels,
        trainset,
        trainable_agg_params,
        require_argmax,
    )
    results["linear_mapping"] = {
        "mse_loss": lm_performance,
        "downstream_train_accuracy": -1,
        "downstream_test_accuracy": -1,
    }

    proxy_latents, proxy_dataset = common.sample_proxy_dataset(
        models=models, samples=1_000, device=device
    )

    results["neural_network"] = neural.run_and_evaluate(
        agg_params=trainable_agg_params,
        train_loader=trainset,
        test_loader=testset,
        downstream_test_loader=test_loader,
        proxy_dataset_tensor=proxy_dataset,
        num_labels=num_labels,
        device=device,
    )

    results["distillation"] = distillation.run_and_evaluate(
        model_config=trainable_agg_params["model_config"],
        test_loader=test_loader,
        proxy_latents_tensor=proxy_latents,
        proxy_dataset_tensor=proxy_dataset,
        num_labels=num_labels,
        device=device,
    )

    table = wandb.Table(columns=["Aggregation", "Performance"])
    for k, v in results.items():
        table.add_data(k, v)

    wandb.log({"Aggregation Performance": table})

    return results
