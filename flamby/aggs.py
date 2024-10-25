import torch
import numpy as np
import logging
from train import train_and_evaluate
import wandb

def evaluate_competencies(i, dataset, total_clients, num_classes, require_argmax=False):

    total_correct = 0
    total_predicted = 0

    competency_matrix = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    total_samples_per_class = [0.0 for _ in range(num_classes)]

    for X, y in dataset:
        if require_argmax: # multiclass classification
            # X: (batch_size, total_clients * num_classes)
            X = torch.reshape(X, (-1, total_clients, num_classes))
            output = X[:, i, :]
            _, predictions = torch.max(output, 1)
        else: # binary classification
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
    logging.debug(f"Total seen classes {len(seen_classes)} and unseen classes {len(unseen_classes)}")
    
    for k in seen_classes:
        for j in seen_classes:
            competency_matrix[k][j] /= total_samples_per_class[k]

    for k in unseen_classes:
        for j in seen_classes:
            competency_matrix[k][j] = 1.0/len(seen_classes)

    accuracy = total_correct/total_predicted
    logging.debug("Competence Accuracy: {:.4f}".format(accuracy))
    return competency_matrix

def get_prediction_using_competency(competencies, num_classes):
    classwise_estimates = []
    for m in range(num_classes):
        ans = 1.0
        for node_competency in competencies:
            ans *= node_competency[m]    
        classwise_estimates.append(ans)
    
    confidences = torch.softmax(torch.tensor(classwise_estimates), dim=0)
    return torch.argmax(confidences).item(), confidences

def run_forward_linearagg(weights, bias, criterion, testset, total_clients, 
                          num_classes, metric, require_argmax=False):
        y_preds = []
        y_trues = []
        loss = 0.0
        n_batches = len(testset)
        reshape_dim = 1 if not require_argmax else num_classes

        with torch.no_grad():
            for X, y in testset:
                X = torch.reshape(X, (-1, total_clients, reshape_dim))
                X = torch.swapaxes(X, 1, 2)
                X = X*weights
                y_pred = torch.sum(X, dim=2) + bias
                target = y.reshape(y_pred.shape) if not require_argmax else y
                loss += criterion(y_pred, target).detach().cpu().item()
                if require_argmax:
                    y_pred = y_pred.argmax(dim=1)
                
                y_preds.append(y_pred.detach().cpu())
                y_trues.append(y)

        lm_loss = loss / n_batches
        
        y_preds_np = np.concatenate(y_preds)
        y_preds_np = y_preds_np.squeeze(-1) if y_preds_np.shape[-1] == 1 else y_preds_np

        y_trues_np = np.concatenate(y_trues)
        y_trues_np = y_trues_np.squeeze(-1) if y_trues_np.shape[-1] == 1 else y_trues_np

        lm_performance = metric(y_trues_np, y_preds_np)
        return lm_loss, lm_performance

def averaging(dataset, metric, total_clients, num_classes, require_argmax=False):
    y_preds = []
    y_trues = []

    for X, y in dataset:
        if require_argmax:
            # X: (batch_size, total_clients * num_classes)
            X = torch.reshape(X, (-1, total_clients, num_classes))
            y_pred = torch.mean(X, dim=1).argmax(dim=1)
        else:
            # X: (batch_size, total_clients)
            X = torch.sigmoid(X)
            y_pred = torch.mean(X, dim=1)
            y_pred = torch.log(y_pred / (1 - y_pred))
        
        y_preds.append(y_pred)
        y_trues.append(y)

    y_preds_np = np.concatenate(y_preds)
    y_preds_np = y_preds_np.squeeze(-1) if y_preds_np.shape[-1] == 1 else y_preds_np

    y_trues_np = np.concatenate(y_trues)
    y_trues_np = y_trues_np.squeeze(-1) if y_trues_np.shape[-1] == 1 else y_trues_np    

    avg_performance = metric(y_trues_np, y_preds_np)
    logging.info(f'==> Averaging Performance: {avg_performance:.4f}')
    
    return avg_performance

def weighted_averaging(dataset, metric, total_clients, num_classes, label_dists, require_argmax=False):
    # Get weights from labels
    my_labels_tensor = torch.tensor(label_dists) # (num_clients, num_classes)
    label_sum_tensor = my_labels_tensor.sum(dim=0) # (num_classes)
    my_weights_tensor = my_labels_tensor / label_sum_tensor # (num_clients, num_classes)

    y_preds = []
    y_trues = []
    for X, y in dataset:
        if require_argmax:
            # X: (batch_size, total_clients * num_classes)
            X = torch.reshape(X, (-1, total_clients, num_classes))
            X = X*my_weights_tensor
            y_pred = torch.sum(X, dim=1).argmax(dim=1)
        else:
            # X: (batch_size, total_clients)
            X = torch.sigmoid(X)
            X_complement = 1 - X
            X_all = torch.stack((X_complement, X), dim=2) # (batch_size, total_clients, 2)
            X = X_all*my_weights_tensor # (batch_size, total_clients, 2)
            y_avg = torch.sum(X, dim=1)
            y_avg = torch.softmax(y_avg, dim=1)
            y_pred = torch.log(y_avg[:, 1] / y_avg[:, 0])

        y_preds.append(y_pred)
        y_trues.append(y)

    y_preds_np = np.concatenate(y_preds)
    y_preds_np = y_preds_np.squeeze(-1) if y_preds_np.shape[-1] == 1 else y_preds_np

    y_trues_np = np.concatenate(y_trues)
    y_trues_np = y_trues_np.squeeze(-1) if y_trues_np.shape[-1] == 1 else y_trues_np

    wavg_performance = metric(y_trues_np, y_preds_np)
    logging.info(f'==> Weighted Averaging Performance: {wavg_performance:.4f}')

    return wavg_performance
    
def polychotomous_voting(dataset, metric, total_clients, num_classes, models, device, train_dataset, require_argmax=False):
    competency_matrices = {}
    for i in range(len(models)):
            cm = evaluate_competencies(i, train_dataset, total_clients,
                                       num_classes, require_argmax)
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
                relevant_compt_for_client = [competency_matrices[client_idx][i][pred_class_by_client] for i in range(num_classes)]
                relevant_competencies.append(relevant_compt_for_client)
            
            y_pred_for_elem_idx, confidences = \
                get_prediction_using_competency(relevant_competencies, num_classes)
            
            if not require_argmax:
                y_pred_for_elem_idx = torch.log(confidences[1] / confidences[0]).item()

            y_preds.append(y_pred_for_elem_idx)
        y_trues.append(y)
        
    y_preds_np = np.array(y_preds)
    y_preds_np = y_preds_np.squeeze(-1) if y_preds_np.shape[-1] == 1 else y_preds_np

    y_trues_np = np.concatenate(y_trues)
    y_trues_np = y_trues_np.squeeze(-1) if y_trues_np.shape[-1] == 1 else y_trues_np

    voting_performance = metric(y_trues_np, y_preds_np)
    logging.info(f'==> Voting Performance: {voting_performance}')
    
    return voting_performance

def linear_mapping(dataset, metric, total_clients, num_classes, train_dataset, agg_params, require_argmax=False):
    id_str = 'lm_agg'
    wandb.define_metric(f'{id_str}/epoch')
    wandb.define_metric(f'{id_str}/*', step_metric=f'{id_str}/epoch')

    weights = torch.ones(total_clients, requires_grad=True, dtype=torch.float64)
    logging.debug("Size of weights {}".format(weights.size()))
    bias = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
    logging.debug("Size of bias {}".format(bias.size()))

    lr = agg_params['lm_lr']
    epochs = agg_params['lm_epochs']    
    criterion = agg_params['criterion']
    
    optimizer = torch.optim.Adam([weights, bias], lr=lr)
    n_batches = len(train_dataset)

    reshape_dim = 1 if not require_argmax else num_classes

    y_preds = []
    y_trues = []
    best_acc = 0.0
    for e in range(epochs):
        epoch_loss = 0.0
        
        for X, y in train_dataset:
            optimizer.zero_grad()
            X = torch.reshape(X, (-1, total_clients, reshape_dim))
            X = torch.swapaxes(X, 1, 2)
            X = X*weights
            y_pred = torch.sum(X, dim=2) + bias
            target = y.reshape(y_pred.shape) if not require_argmax else y            
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss +=  loss.detach().cpu().item()
            
            if require_argmax:
                y_pred = y_pred.argmax(dim=1)

            y_preds.append(y_pred.detach().cpu())
            y_trues.append(y)            
        
        avg_epoch_loss = epoch_loss / n_batches

        y_preds_np = torch.cat(y_preds).numpy()
        y_preds_np = y_preds_np.squeeze(-1) if y_preds_np.shape[-1] == 1 else y_preds_np

        y_trues_np = torch.cat(y_trues).numpy()
        y_trues_np = y_trues_np.squeeze(-1) if y_trues_np.shape[-1] == 1 else y_trues_np

        acc = metric(y_trues_np, y_preds_np)    
        logging.info(f'Epoch {e+1} Train Loss {avg_epoch_loss:.4f} Train Acc {acc:.4f}')
        wandb.log({f'{id_str}/train_loss': avg_epoch_loss, f'{id_str}/train_acc': acc,
                   f'{id_str}/epoch': e+1})
        
        if e%10 == 0:
            lm_loss, lm_performance = \
                run_forward_linearagg(weights, bias, criterion, 
                                      dataset, total_clients, num_classes, 
                                      metric, require_argmax)
            logging.info(f'Epoch {e+1} Test Loss {lm_loss:.4f} Test Acc {lm_performance:.4f}')
            wandb.log({f'{id_str}/test_loss': lm_loss, f'{id_str}/test_acc': lm_performance,
                       f'{id_str}/epoch': e+1})
            if(lm_performance > best_acc):
                logging.debug(f'Improved performance from {best_acc:.4f} to {lm_performance:.4f}')
                best_acc = lm_performance
                
    logging.info(f'==> Best Linear Mapping Performance: {best_acc}')
    
    return best_acc

def nn_mapping(dataset, metric, train_dataset, device, agg_params, require_argmax=False):    
    id_str = 'nn_agg'
    f = agg_params["nn_model"]()
    loss = agg_params["criterion"]
    lr = agg_params["nn_lr"]
    epochs = agg_params["nn_epochs"]
    optimizer = torch.optim.Adam(f.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    nn_performance, _ = train_and_evaluate(id_str, f, loss, optimizer, scheduler, train_dataset, 
        dataset, epochs, device, metric, test_every=5, require_argmax=require_argmax)
    
    logging.info(f'==> Best NN Performance: {nn_performance}')
    
    return nn_performance

def evaluate_all_aggregations(train_loader, test_loader, models, label_dists, metric, 
                              device, trainable_agg_params, require_argmax=False):
    # elems: (batch_size, input_dim)
    # model(elems): (batch_size, output_dim)
    # hstack(outputs): (batch_size, output_dim * num_models)

    # move models to device
    for model in models:
        model = model.to(device)

    # Create the dataset of predictions for training and testing
    trainset = []
    with torch.no_grad():
        for elems, labels in train_loader:
            elems = elems.to(device)
            outputs = [model(elems).detach().cpu() for model in models]
            stacked_outputs = torch.hstack(outputs)
            trainset.append((stacked_outputs, labels))

    testset = []
    with torch.no_grad():
        for elems, labels in test_loader:
            elems = elems.to(device)
            outputs = [model(elems).detach().cpu() for model in models]
            stacked_outputs = torch.hstack(outputs)
            testset.append((stacked_outputs, labels))
    
    results = {}

    avg_performance = averaging(testset, metric, len(models), len(label_dists[0]), require_argmax)
    results['avg'] = avg_performance

    wavg_performance = weighted_averaging(testset, metric, len(models), len(label_dists[0]), label_dists, require_argmax)
    results['wavg'] = wavg_performance

    voting_performance = polychotomous_voting(testset, metric, len(models), len(label_dists[0]), models, device, trainset, require_argmax)
    results['voting'] = voting_performance

    lm_performance = linear_mapping(testset, metric, len(models), len(label_dists[0]), trainset, trainable_agg_params, require_argmax)
    results['linear_mapping'] = lm_performance

    nn_performance = nn_mapping(testset, metric, trainset, device, trainable_agg_params, require_argmax)
    results['neural_network'] = nn_performance

    table = wandb.Table(columns=["Aggregation", "Performance"])
    for k, v in results.items():
        table.add_data(k, v)

    wandb.log({"Aggregation Performance": table})

    return results