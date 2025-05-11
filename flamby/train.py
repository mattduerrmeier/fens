import logging
import copy
import torch
import wandb
import math
from autoencoder.model import Autoencoder
from tqdm import tqdm


def train_and_evaluate_aggs(
    id_str,
    model,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    n_epochs,
    device,
) -> tuple[float, torch.nn.Module]:
    """
    Updated version of the training for the VAE model.
    No downstream metric for the vae, so we only log the loss.
    """
    wandb.define_metric(f"{id_str}/epoch")
    wandb.define_metric(f"{id_str}/*", step_metric=f"{id_str}/epoch")

    model = model.to(device)

    best_model = None
    epoch = 0
    best_loss = math.inf
    for epoch in range(1, n_epochs + 1):
        loss_acc = 0.0
        mse_acc, kld_acc = 0.0, 0.0

        model.train()
        for data, target in train_loader:
            data = data.swapaxes(0, 1)
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # forward pass
            out = model(data)
            # need to extract the two term to be able to log both of them
            loss = loss_fn(out, target)

            # backward pass
            loss.backward()
            optimizer.step()

            # TODO: why is there a coeff here?
            loss_acc += loss.item()  # * data.size(0)

        scheduler.step()

        train_loss = loss_acc / len(train_loader)

        logging.info(f"Epoch: {epoch}, train loss: {train_loss:.4f}")
        wandb.log(
            {
                f"{id_str}/train_loss": train_loss,
                f"{id_str}/epoch": epoch,
            }
        )

        model.eval()
        test_loss = evaluate_agg(model, loss_fn, test_loader, device)
        if test_loss < best_loss:
            best_loss = test_loss
            # store the best model by making a copy
            best_model = copy.deepcopy(model)

        logging.info(f"Epoch: {epoch} test loss: {test_loss:.4f}")
        wandb.log(
            {
                f"{id_str}/test_loss": test_loss,
                f"{id_str}/epoch": epoch,
            }
        )

    return best_loss, best_model


def evaluate_agg(model, loss_fn, test_loader, device):
    model.eval()
    model.to(device)

    loss_acc = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            # fwd pass
            data = data.swapaxes(0, 1)
            data = data.to(device)
            target = target.to(device)

            out = model(data)
            loss = loss_fn(out, target)

            loss_acc += loss.item()  # * data.size(0)

    loss_acc /= len(test_loader)
    return loss_acc


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

    model = model.to(device)

    best_model = None
    epoch = 0
    best_loss = math.inf

    for epoch in range(1, n_epochs + 1):
        loss_acc = 0.0
        mse_acc, kld_acc = 0.0, 0.0

        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.unsqueeze(dim=1).to(device)

            optimizer.zero_grad()

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

        model.eval()
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

    loss_acc /= len(test_loader)
    mse_acc /= len(test_loader)
    kld_acc /= len(test_loader)

    return loss_acc, mse_acc, kld_acc


def _determine_dataset_feature_count(loader: torch.utils.data.DataLoader) -> int:
    batch_features, _batch_targets = next(iter(loader))
    return batch_features.shape[1]


class _DownstreamTaskModel(torch.nn.Module):
    def __init__(self, input_dimensions: int):
        super(_DownstreamTaskModel, self).__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Sequential(
                torch.nn.Linear(input_dimensions, 64),
                torch.nn.LeakyReLU(),
            ),
            torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.LeakyReLU()),
            torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.LeakyReLU(),
            ),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


def evaluate_downstream_task(
    id: str,
    loader_train: torch.utils.data.DataLoader,
    loader_test: torch.utils.data.DataLoader,
) -> tuple[float, float]:
    input_dimensions = _determine_dataset_feature_count(loader_train)

    model = _DownstreamTaskModel(input_dimensions)
    optimizer = torch.optim.AdamW(model.parameters())

    loss_function = torch.nn.CrossEntropyLoss()

    def train() -> tuple[float, float]:
        model.train()

        losses: list[float] = []
        correct_predictions = 0
        total_predictions = 0

        for features_train, targets_train in loader_train:
            targets_train = targets_train.squeeze(dim=1)

            optimizer.zero_grad()

            outputs_train = model(features_train)
            loss = loss_function(outputs_train, targets_train.long())

            loss.backward()
            optimizer.step()

            predictions_train = torch.argmax(outputs_train, 1)

            losses.append(loss.item())
            correct_predictions += (predictions_train == targets_train).sum().item()
            total_predictions += predictions_train.shape[0]

        return (
            torch.tensor(losses).mean().item(),
            correct_predictions / total_predictions,
        )

    def evaluate() -> tuple[float, float]:
        model.eval()

        losses: list[float] = []
        correct_predictions = 0
        total_predictions = 0

        for features_test, targets_test in loader_test:
            targets_test = targets_test.squeeze(dim=1)

            outputs_test = model(features_test)

            loss = loss_function(outputs_test, targets_test.long())

            predictions_test = torch.argmax(outputs_test, 1)

            losses.append(loss.item())
            correct_predictions += (predictions_test == targets_test).sum().item()
            total_predictions += predictions_test.shape[0]

        return (
            torch.tensor(losses).mean().item(),
            correct_predictions / total_predictions,
        )

    train_accuracy = 0
    test_accuracy = 0

    for epoch in range(1, 10):
        train_loss, train_accuracy = train()
        test_loss, test_accuracy = evaluate()

        wandb.log(
            {
                f"{id}/downstream_train_loss": train_loss,
                f"{id}/downstream_train_accuracy": train_accuracy,
                f"{id}/downstream_test_loss": test_loss,
                f"{id}/downstream_test_accuracy": test_accuracy,
                f"{id}/epoch": epoch,
            }
        )
        print(
            "Downstream task training: "
            "epoch {}, train loss: {:.2f}, train acc:{:.2f}%, test loss: {:.2f}, test acc: {:.2f}%".format(
                epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100
            )
        )

    return train_accuracy, test_accuracy
