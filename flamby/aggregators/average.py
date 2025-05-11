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
        # if require_argmax:
        #     # out: (batch_size, total_clients * num_classes)
        #     out = torch.reshape(out, (-1, total_clients, num_classes))
        #     out = torch.mean(out, dim=0).argmax(dim=1)
        # else:
        #     # out: (batch_size, total_clients)
        #     out = torch.sigmoid(out)
        #     out = torch.log(out / (1 - out))

        # y_preds.append(y_pred)
        # y_trues.append(y)
        inputs.append(data)
        outputs.append(out)

    inputs = torch.cat(inputs)
    outputs = torch.cat(outputs)

    print("inputs", inputs.shape)
    print("outputs", outputs.shape)

    avg_performance = metric(outputs, inputs).item()
    logging.info(f"==> Averaging Performance: {avg_performance:.4f}")

    return avg_performance
