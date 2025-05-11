import logging

import torch
from train import train_and_evaluate_aggs


def nn_mapping(
    dataset, metric, train_dataset, device, agg_params, require_argmax=False
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
