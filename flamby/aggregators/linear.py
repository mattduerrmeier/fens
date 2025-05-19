import logging
import torch
import wandb


def run_forward_linearagg(
    weights,
    bias,
    criterion,
    testset,
    total_clients,
    num_classes,
    metric,
    require_argmax=False,
):
    inputs = []
    outputs = []

    loss = 0.0
    n_batches = len(testset)
    reshape_dim = 1 if not require_argmax else num_classes

    with torch.no_grad():
        for out, data in testset:
            out = (out.permute(*torch.arange(out.ndim - 1, -1, -1)) @ weights) + bias
            out = out.permute(*torch.arange(out.ndim - 1, -1, -1))

            loss = criterion(out, data)

            inputs.append(data)
            outputs.append(out.detach().cpu())

    lm_loss = loss / n_batches

    inputs = torch.cat(inputs)
    outputs = torch.cat(outputs)

    lm_performance = metric(inputs, outputs)
    return lm_loss, lm_performance.item()


def linear_mapping(
    dataset,
    metric,
    total_clients,
    num_classes,
    train_dataset,
    agg_params,
    require_argmax=False,
):
    id_str = "lm_agg"
    wandb.define_metric(f"{id_str}/epoch")
    wandb.define_metric(f"{id_str}/*", step_metric=f"{id_str}/epoch")

    weights = torch.ones(total_clients, requires_grad=True, dtype=torch.float32)
    bias = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    logging.info("Size of weights: {}".format(weights.size()))
    logging.info("Size of bias: {}".format(bias.size()))

    lr = agg_params["lm_lr"]
    epochs = agg_params["lm_epochs"]
    criterion = agg_params["criterion"]

    optimizer = torch.optim.Adam([weights, bias], lr=lr)
    n_batches = len(train_dataset)

    reshape_dim = 1 if not require_argmax else num_classes

    best_acc = float("inf")
    for epoch in range(epochs):
        inputs = []
        outputs = []
        epoch_loss = 0.0

        # for out, y in train_dataset:
        for out, data in train_dataset:
            optimizer.zero_grad()
            out = (out.permute(*torch.arange(out.ndim - 1, -1, -1)) @ weights) + bias
            out = out.permute(*torch.arange(out.ndim - 1, -1, -1))

            # target = y.reshape(y_pred.shape) if not require_argmax else y
            loss = criterion(out, data)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().item()

            if require_argmax:
                y_pred = y_pred.argmax(dim=1)

            inputs.append(data)
            outputs.append(out.detach().cpu())

        avg_epoch_loss = epoch_loss / n_batches

        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)

        train_performance = metric(inputs, outputs)
        logging.info(
            f"Epoch {epoch + 1}, train loss {avg_epoch_loss:.4f}, train mse {train_performance:.4f}"
        )
        wandb.log(
            {
                f"{id_str}/train_loss": avg_epoch_loss,
                f"{id_str}/train_mse": train_performance,
                f"{id_str}/epoch": epoch + 1,
            }
        )

        if epoch % 10 == 0:
            lm_loss, lm_performance = run_forward_linearagg(
                weights,
                bias,
                criterion,
                dataset,
                total_clients,
                num_classes,
                metric,
                require_argmax,
            )
            logging.info(
                f"Epoch {epoch + 1}, test loss {lm_loss:.4f}, test mse {lm_performance:.4f}"
            )
            wandb.log(
                {
                    f"{id_str}/test_loss": lm_loss,
                    f"{id_str}/test_mse": lm_performance,
                    f"{id_str}/epoch": epoch + 1,
                }
            )
            if lm_performance < best_acc:
                logging.debug(
                    f"Improved performance from {best_acc:.4f} to {lm_performance:.4f}"
                )
                best_acc = lm_performance

    logging.info(f"==> Best Linear Mapping Performance: {best_acc}")

    return best_acc
