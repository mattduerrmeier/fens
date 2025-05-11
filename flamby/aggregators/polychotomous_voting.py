import logging

import numpy as np
import torch


def _evaluate_competencies(
    i, dataset, total_clients, num_classes, require_argmax=False
):
    total_correct = 0
    total_predicted = 0

    competency_matrix = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    total_samples_per_class = [0.0 for _ in range(num_classes)]

    for X, y in dataset:
        if require_argmax:  # multiclass classification
            # X: (batch_size, total_clients * num_classes)
            X = torch.reshape(X, (-1, total_clients, num_classes))
            output = X[:, i, :]
            _, predictions = torch.max(output, 1)
        else:  # binary classification
            # X: (batch_size, total_clients)
            output = X[:, i]
            predictions = torch.round(torch.sigmoid(output))
        target = y

        for ground_truth, prediction in zip(target, predictions):
            prediction = int(prediction.item())
            ground_truth = int(ground_truth.item())
            competency_matrix[ground_truth][prediction] += 1.0
            total_samples_per_class[ground_truth] += 1.0
            if ground_truth == prediction:
                total_correct += 1
            total_predicted += 1

    seen_classes = [k for k in range(num_classes) if total_samples_per_class[k] != 0]
    unseen_classes = [k for k in range(num_classes) if total_samples_per_class[k] == 0]
    logging.debug(
        f"Total seen classes {len(seen_classes)} and unseen classes {len(unseen_classes)}"
    )

    for k in seen_classes:
        for j in seen_classes:
            competency_matrix[k][j] /= total_samples_per_class[k]

    for k in unseen_classes:
        for j in seen_classes:
            competency_matrix[k][j] = 1.0 / len(seen_classes)

    accuracy = total_correct / total_predicted
    logging.debug("Competence Accuracy: {:.4f}".format(accuracy))
    return competency_matrix


def _get_prediction_using_competency(competencies, num_classes):
    classwise_estimates = []
    for m in range(num_classes):
        ans = 1.0
        for node_competency in competencies:
            ans *= node_competency[m]
        classwise_estimates.append(ans)

    confidences = torch.softmax(torch.tensor(classwise_estimates), dim=0)
    return torch.argmax(confidences).item(), confidences


def polychotomous_voting(
    dataset,
    metric,
    total_clients,
    num_classes,
    models,
    device,
    train_dataset,
    require_argmax=False,
):
    competency_matrices = {}
    for i in range(len(models)):
        cm = _evaluate_competencies(
            i, train_dataset, total_clients, num_classes, require_argmax
        )
        competency_matrices[i] = cm

    y_preds = []
    y_trues = []
    for X, y in dataset:
        if require_argmax:
            # X: (batch_size, total_clients * num_classes)
            X = torch.reshape(X, (-1, total_clients, num_classes))
            X = torch.argmax(X, 2)
        else:
            # X: (batch_size, total_clients)
            X = torch.round(torch.sigmoid(X)).int()
        for elem_idx in range(X.size(0)):
            relevant_competencies = []
            for client_idx in range(X.size(1)):
                pred_class_by_client = X[elem_idx][client_idx]
                relevant_compt_for_client = [
                    competency_matrices[client_idx][i][pred_class_by_client]
                    for i in range(num_classes)
                ]
                relevant_competencies.append(relevant_compt_for_client)

            y_pred_for_elem_idx, confidences = _get_prediction_using_competency(
                relevant_competencies, num_classes
            )

            if not require_argmax:
                y_pred_for_elem_idx = torch.log(confidences[1] / confidences[0]).item()

            y_preds.append(y_pred_for_elem_idx)
        y_trues.append(y)

    y_preds_np = np.array(y_preds)
    y_preds_np = y_preds_np.squeeze(-1) if y_preds_np.shape[-1] == 1 else y_preds_np

    y_trues_np = np.concatenate(y_trues)
    y_trues_np = y_trues_np.squeeze(-1) if y_trues_np.shape[-1] == 1 else y_trues_np

    voting_performance = metric(y_trues_np, y_preds_np)
    logging.info(f"==> Voting Performance: {voting_performance}")

    return voting_performance
