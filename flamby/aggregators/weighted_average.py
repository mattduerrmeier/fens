import logging
import torch


def weighted_averaging(
    dataset, metric, total_clients, num_classes, label_dists, require_argmax=False
):
    # Get weights from labels
    my_labels_tensor = torch.tensor(label_dists)  # (num_clients, num_classes)
    label_sum_tensor = my_labels_tensor.sum(dim=0)  # (num_classes)
    my_weights_tensor = (
        my_labels_tensor / label_sum_tensor
    )  # (num_clients, num_classes)

    inputs = []
    outputs = []
    for out, data in dataset:
        local_weight = my_weights_tensor[:, labels.int()]
        out = out * local_weight
        out = torch.sum(out, dim=0)
        # if require_argmax:
        #     # out: (batch_size, total_clients * num_classes)
        #     out = torch.reshape(out, (-1, total_clients, num_classes))
        #     out = out * my_weights_tensor
        #     y_pred = torch.sum(out, dim=1).argmax(dim=1)
        # else:
        #     # out: (batch_size, total_clients)
        #     out = torch.sigmoid(out)
        #     out_complement = 1 - out
        #     out_all = torch.stack(
        #         (out_complement, out), dim=2
        #     )  # (batch_size, total_clients, 2)
        #     out = out_all * my_weights_tensor  # (batch_size, total_clients, 2)
        #     out = torch.sum(out, dim=1)
        #     out = torch.softmax(out, dim=1)
        #     out = torch.log(out[:, 1] / out[:, 0])

        inputs.append(data)
        outputs.append(out)

    inputs = torch.cat(inputs)
    outputs = torch.cat(outputs)

    wavg_performance = metric(outputs, inputs).item()
    logging.info(f"==> Weighted Averaging Performance: {wavg_performance:.4f}")

    return wavg_performance
