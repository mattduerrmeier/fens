import logging
import torch


def weighted_averaging(
    dataset, metric, total_clients, num_classes, label_dists, require_argmax=False
):
    # Get weights from labels
    labels_tensor = torch.tensor(label_dists)
    label_sum_tensor = labels_tensor.sum(dim=0)
    weighted_label_tensor = labels_tensor / label_sum_tensor

    inputs = []
    outputs = []
    for out, data in dataset:
        if num_classes > 2:
            labels = out[:, :, -num_classes:].argmax(dim=-1)
        else:
            labels = out[:, :, -1]

        local_weight = torch.take_along_dim(weighted_label_tensor, labels.long(), dim=1)
        out = out * local_weight.unsqueeze(dim=-1)
        out = torch.sum(out, dim=0)

        inputs.append(data)
        outputs.append(out)

    inputs = torch.cat(inputs)
    outputs = torch.cat(outputs)

    wavg_performance = metric(outputs, inputs).item()
    logging.info(f"==> Weighted Averaging Performance: {wavg_performance:.4f}")

    return wavg_performance
