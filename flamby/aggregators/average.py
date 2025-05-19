import logging
import torch


def averaging(dataset, metric, total_clients, num_classes, require_argmax=False):
    y_preds = []
    y_trues = []

    # out from the model, target is the original X
    inputs = []
    outputs = []
    for out, data in dataset:
        out = torch.mean(out, dim=0)

        inputs.append(data)
        outputs.append(out)

    inputs = torch.cat(inputs)
    outputs = torch.cat(outputs)

    avg_performance = metric(outputs, inputs).item()
    logging.info(f"==> Averaging Performance: {avg_performance:.4f}")

    return avg_performance
