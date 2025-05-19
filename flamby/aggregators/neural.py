import logging
import typing
import torch
from train import evaluate_downstream_task, train_and_evaluate_aggs

from .common import AggregatorResult


def run_and_evaluate(
    agg_params: typing.Mapping[str, typing.Any],
    train_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    downstream_test_loader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor]
    ],
    proxy_dataset_tensor: torch.Tensor,
    num_labels: int,
    device: torch.device,
) -> AggregatorResult:
    mse_loss, best_model = nn_mapping(
        agg_params=agg_params,
        train_dataset=train_loader,
        dataset=test_loader,
        metric=None,
        device=device,
    )

    train_accuracy, test_accuracy = evaluate_on_downstream_task(
        best_model, proxy_dataset_tensor, downstream_test_loader, num_labels, device
    )

    return {
        "mse_loss": mse_loss,
        "downstream_train_accuracy": train_accuracy,
        "downstream_test_accuracy": test_accuracy,
    }


def nn_mapping(
    agg_params, train_dataset, dataset, metric, device
) -> tuple[float, torch.nn.Module]:
    id_str = "nn_agg"
    f = agg_params["nn_model"]()
    loss = agg_params["criterion"]
    lr = agg_params["nn_lr"]
    epochs = agg_params["nn_epochs"]
    optimizer = torch.optim.Adam(f.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    nn_performance, best_model = train_and_evaluate_aggs(
        id_str,
        f,
        loss,
        optimizer,
        scheduler,
        train_dataset,
        dataset,
        epochs,
        device,
    )

    logging.info(f"==> Best NN Performance: {nn_performance}")

    # TODO: Replace with value for actual metric (and not loss, as here)
    return nn_performance, best_model


def evaluate_on_downstream_task(
    model: torch.nn.Module,
    proxy_dataset: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
    num_labels: int,
    device: torch.device,
) -> tuple[float, float]:
    proxy_dataset = proxy_dataset.swapaxes(0, 1)
    downstream_dataset = model(proxy_dataset).detach()

    synthetic_x: torch.Tensor
    synthetic_y: torch.Tensor
    if num_labels > 2:
        synthetic_x = downstream_dataset[:, :-num_labels]
        synthetic_y = (
            downstream_dataset[:, -num_labels:]
            .clip(0, 1)
            .round()
            .argmax(dim=1, keepdim=True)
        )
    else:
        synthetic_x = downstream_dataset[:, :-1]
        synthetic_y = downstream_dataset[:, -1].unsqueeze(dim=1).clip(0, 1).round()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(synthetic_x, synthetic_y),
        batch_size=32,
    )

    train_accuracy, test_accuracy = evaluate_downstream_task(
        "nn_agg",
        train_loader,
        test_loader,
        num_labels,
        device,
    )

    return train_accuracy, test_accuracy
