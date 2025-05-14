import torch
from aggregators.average import averaging
from aggregators.distillation import train_student
from aggregators.distillation import (
    evaluate_on_downstream_task as evaluate_distillation_on_downstream_task,
)
from aggregators.linear import linear_mapping
from aggregators.neural import (
    evaluate_on_downstream_task as evaluate_neural_on_downstream_task,
)
from aggregators.neural import nn_mapping
from autoencoder.model import Autoencoder, Decoder
from train import evaluate_downstream_task

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
            data = torch.concat((elems, labels), dim=1)

            proxy_dataset.append((stacked_outputs, data))

    return proxy_dataset


def _sample_proxy_dataset(
    models: list[Autoencoder | Decoder],
    samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    for model in models:
        latent = model.sample_latent(samples)
        synthetic_x, synthetic_y = model.sample_from_latent(latent)

        latents.append(latent)
        outputs.append(
            torch.cat(
                (synthetic_x.detach(), synthetic_y.detach().clip(0, 1).round()), dim=1
            )
        )

    return torch.stack(latents), torch.stack(outputs)


def evaluate_all_aggregations(
    train_loader,
    test_loader,
    models,
    label_dists,
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
        testset, metric, len(models), len(label_dists[0]), require_argmax
    )
    results["avg"] = {
        "mse_loss": avg_performance,
        "downstream_train_accuracy": -1,
        "downstream_test_accuracy": -1,
    }

    if False:
        wavg_performance = weighted_averaging(
            testset,
            metric,
            len(models),
            len(label_dists[0]),
            label_dists,
            require_argmax,
        )
        results["wavg"] = wavg_performance

    if False:
        voting_performance = polychotomous_voting(
            testset,
            metric,
            len(models),
            len(label_dists[0]),
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
        len(label_dists[0]),
        trainset,
        trainable_agg_params,
        require_argmax,
    )
    results["linear_mapping"] = {
        "mse_loss": lm_performance,
        "downstream_train_accuracy": -1,
        "downstream_test_accuracy": -1,
    }

    nn_performance, best_model = nn_mapping(
        testset, metric, trainset, device, trainable_agg_params, require_argmax
    )

    proxy_latents, proxy_dataset = _sample_proxy_dataset(models=models, samples=1_000)

    nn_agg_train_accuracy, nn_agg_test_accuracy = evaluate_neural_on_downstream_task(
        best_model, proxy_dataset, test_loader
    )

    results["neural_network"] = {
        "mse_loss": nn_performance,
        "downstream_train_accuracy": nn_agg_train_accuracy,
        "downstream_test_accuracy": nn_agg_test_accuracy,
    }

    # TODO: determine number of features dynamically
    student_model = Decoder(output_dimensions=13 + 1)
    best_student_model = train_student(
        student_model=student_model,
        proxy_dataset=torch.utils.data.TensorDataset(
            proxy_latents.flatten(end_dim=1),
            proxy_dataset.flatten(end_dim=1)[:, :-1],
            proxy_dataset.flatten(end_dim=1)[:, -1:],
        ),
        epochs=200,
        batch_size=64,
        optimizer=torch.optim.Adam(student_model.parameters(), lr=1e-3),
    )

    _, downstream_proxy_data = _sample_proxy_dataset([best_student_model], 1000)
    evaluate_distillation_on_downstream_task(downstream_proxy_data, test_loader)

    table = wandb.Table(columns=["Aggregation", "Performance"])
    for k, v in results.items():
        table.add_data(k, v)

    wandb.log({"Aggregation Performance": table})

    return results
