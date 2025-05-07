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
    for epoch in range(1, n_epochs + 1):
        loss_acc = 0.0
        mse_acc, kld_acc = 0.0, 0.0

        for data, target in train_loader:
            data = data.to(device)
            target = target.unsqueeze(dim=1).to(device)

            # forward pass
            # TODO: do we pass the target with the data?
            out, mu, logvar = model((data, target))
            # need to extract the two term to be able to log both of them

            concatenated_input = torch.concat((data, target), dim=1)
            mse_loss, kld_loss = loss_fn(out, concatenated_input, mu, logvar)
            loss = mse_loss + kld_loss

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # TODO: why is there a coeff here?
            loss_acc += loss.item()  # * data.size(0)
            mse_acc += mse_loss.item()
            kld_acc += kld_loss.item()

        scheduler.step()

        train_loss = loss_acc / len(train_loader)
        train_mse = mse_acc / len(train_loader)
        train_kld = kld_acc / len(train_loader)

        logging.info(
            f"Epoch: {epoch}, train loss: {train_loss:.4f}, mse: {train_mse:.4f}, kld: {train_kld:.4f}"
        )
        wandb.log(
            {
                f"{id_str}/train_loss": train_loss,
                f"{id_str}/train_mse": train_mse,
                f"{id_str}/train_kld": train_kld,
                f"{id_str}/epoch": epoch,
            }
        )

        test_loss, test_mse, test_kld = evaluate(model, loss_fn, test_loader, device)
        if test_loss < best_loss:
            best_loss = test_loss
            # store the best model by making a copy
            best_model = copy.deepcopy(model)

        logging.info(
            f"Epoch: {epoch} test loss: {test_loss:.4f}, mse: {test_mse:.4f}, kld: {test_kld:.4f}"
        )
        wandb.log(
            {
                f"{id_str}/test_loss": test_loss,
                f"{id_str}/test_mse": test_mse,
                f"{id_str}/test_kld": test_kld,
                f"{id_str}/epoch": epoch,
            }
        )

    return best_loss, best_model


def evaluate(model, loss_fn, test_loader, device):
    model.eval()
    model.to(device)

    loss_acc = 0.0
    mse_acc, kld_acc = 0.0, 0.0
    counts = 0
    with torch.no_grad():
        for data, target in test_loader:
            # fwd pass
            data = data.to(device)
            out, mu, logvar = model((data, target))

            concatenated_input = torch.concat((data, target), dim=1)
            mse_loss, kld_loss = loss_fn(out, concatenated_input, mu, logvar)

            loss = mse_loss + kld_loss

            loss_acc += loss.item()  # * data.size(0)
            mse_acc += mse_loss.item()
            kld_acc += kld_loss.item()
            counts += data.size(0)

    loss_acc /= counts
    mse_acc /= counts
    kld_acc /= counts
    return loss_acc, mse_acc, kld_acc
