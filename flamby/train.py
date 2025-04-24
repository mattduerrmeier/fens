import logging
import copy
import torch
import wandb
import math


def train_and_evaluate(
    id_str,
    model,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    n_epochs,
    device,
):
    """
    Updated version of the training for the VAE model.
    No downstream metric for the vae, so we only log the loss.
    """
    wandb.define_metric(f"{id_str}/epoch")
    wandb.define_metric(f"{id_str}/*", step_metric=f"{id_str}/epoch")

    model.train()
    model = model.to(device)

    best_model = None
    epoch = 0
    best_loss = math.inf
    for epoch in range(1, n_epochs+1):
        losses = 0
        count = 0
        for data, target in train_loader:
            data = data.to(device)

            # forward pass
            # TODO: do we pass the target with the data?
            out, mu, logvar = model(data)
            loss = loss_fn(out, data, mu, logvar)

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # TODO: why is there a coeff here?
            losses += loss.item()# * data.size(0)
            count += data.size(0)

        scheduler.step()

        losses /= count

        logging.info(f"Epoch: {epoch}, train loss: {losses:.4f}")
        wandb.log({f"{id_str}/train_loss": losses, f"{id_str}/epoch": epoch})

        test_loss = evaluate(model, loss_fn, test_loader, device)
        if test_loss < best_loss:
            best_loss = test_loss
            # store the best model by making a copy
            best_model = copy.deepcopy(model)

        logging.info(f"Epoch: {epoch} test loss: {test_loss:.4f}")
        wandb.log({f"{id_str}/test_loss": test_loss, f"{id_str}/epoch": epoch})

    return best_loss, best_model


def evaluate(model, loss_fn, test_loader, device):
    model.eval()
    model.to(device)

    losses = 0
    counts = 0
    with torch.no_grad():
        for data, target in test_loader:
            # fwd pass
            data = data.to(device)
            out, mu, logvar = model(data)
            loss = loss_fn(out, data, mu, logvar)

            losses += loss.item()# * data.size(0)
            counts += data.size(0)

    losses /= counts
    return losses
