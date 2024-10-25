import torch
from tqdm import tqdm
import numpy as np
import logging

# create a dummy object with two fields: dataset and totalclients
class Dummy:
    def __init__(self, dataset, totalclients):
        self.dataset = dataset
        self.totalclients = totalclients

def load_trainset(client_id, FedDataset, num_classes, proxy_frac=0.1, seed=90):
    trainset = FedDataset(center=client_id, train=True, pooled=False)
    
    t = len(trainset); i = 0
    train_data = {key: [] for key in range(num_classes)}
    for x, y in trainset:
        train_data[int(y.item())].append(x)
        # Somehow the iterator does not know when to terminate
        i += 1
        if(i == t):
            break
    
    rng = np.random.default_rng(seed)
    # shuffle the data
    for key in train_data.keys():
        rng.shuffle(train_data[key])

    # Also store the label distribution
    label_distribution = [0 for _ in range(num_classes)]
    all_trainset = []; all_proxyset = []
    for y, x in train_data.items():
        n_i = len(x) - int(proxy_frac * len(x))
        all_trainset.extend([(a, y) for a in x[:n_i]])
        all_proxyset.extend([(a, y) for a in x[n_i:]])
        label_distribution[y] = n_i

    return all_trainset, all_proxyset, label_distribution

def load_trainset_fedcam16(client_id, FedDataset, num_classes, proxy_frac=0.1, seed=90):
    trainset = FedDataset(center=client_id, train=True, pooled=False)
    
    t = len(trainset); i = 0
    train_data = {key: [] for key in range(num_classes)}
    for x, y, z in trainset:
        train_data[int(y.item())].append((x, z))
        # Somehow the iterator does not know when to terminate
        i += 1
        if(i == t):
            break
    
    rng = np.random.default_rng(seed)
    # shuffle the data
    for key in train_data.keys():
        rng.shuffle(train_data[key])

    # Also store the label distribution
    label_distribution = [0 for _ in range(num_classes)]
    all_trainset = []; all_proxyset = []
    for y, x in train_data.items():
        n_i = len(x) - int(proxy_frac * len(x))
        all_trainset.extend([(a1, torch.tensor(y, dtype=torch.float32), a2) for a1, a2 in x[:n_i]])
        all_proxyset.extend([(a1, torch.tensor(y, dtype=torch.float32), a2) for a1, a2 in x[n_i:]])
        label_distribution[y] = n_i

    return all_trainset, all_proxyset, label_distribution

def load_trainset_combined(dataset_name, client_id, FedDataset, num_classes, proxy_frac=0.1, seed=90):
    trainset = FedDataset(center=client_id, train=True, pooled=False)
    
    t = len(trainset); i = 0
    train_data = {key: [] for key in range(num_classes)}
    if dataset_name == 'FedCamelyon16':
        for x, y, z in trainset:
            train_data[int(y.item())].append((x, z))
            i += 1
            if i == t:
                break
    else:
        for x, y in trainset:
            train_data[int(y.item())].append(x)
            i += 1
            if i == t:
                break
    
    rng = np.random.default_rng(seed)
    for key in train_data.keys():
        rng.shuffle(train_data[key])

    label_distribution = [0 for _ in range(num_classes)]
    all_trainset = []; all_proxyset = []
    for y, x in train_data.items():
        n_i = len(x) - int(proxy_frac * len(x))
        if dataset_name == 'FedCamelyon16':
            all_trainset.extend([(a1, torch.tensor(y, dtype=torch.float32), a2) for a1, a2 in x[:n_i]])
            all_proxyset.extend([(a1, torch.tensor(y, dtype=torch.float32), a2) for a1, a2 in x[n_i:]])
        else:
            all_trainset.extend([(a, y) for a in x[:n_i]])
            all_proxyset.extend([(a, y) for a in x[n_i:]])
        label_distribution[y] = n_i

    return all_trainset, all_proxyset, label_distribution

def generate_logits_combined(dataset_name, trained_models, FedDataset, num_classes, 
        proxy_frac, device, test_dataloader, num_clients, batch_size, seed, collate_fn):
    logits = {}

    trained_models = [model.to(device) for model in trained_models]

    for i in range(num_clients):
        logging.info(f'[Generating logits client {i}]')
        logits[i] = {}

        train_dataset, proxy_dataset, _ = load_trainset_combined(dataset_name, i, FedDataset, num_classes, proxy_frac, seed)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        proxy_dataloader = torch.utils.data.DataLoader(
            proxy_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        
        logit_trainset = []
        with torch.no_grad():
            for elems, labels in train_dataloader:
                elems = elems.to(device)
                outputs = [model(elems).detach().cpu() for model in trained_models]
                stacked_outputs = torch.hstack(outputs)
                logit_trainset.append((stacked_outputs, labels))
        flattened_logit_trainset = []
        for x, y in logit_trainset:
            flattened_logit_trainset.extend(zip(x, y))
        logits[i]['train'] = flattened_logit_trainset

        logit_proxyset = []
        with torch.no_grad():
            for elems, labels in proxy_dataloader:
                elems = elems.to(device)
                outputs = [model(elems).detach().cpu() for model in trained_models]
                stacked_outputs = torch.hstack(outputs)
                logit_proxyset.append((stacked_outputs, labels))
        flattened_logit_proxyset = []
        for x, y in logit_proxyset:
            flattened_logit_proxyset.extend(zip(x, y))
        logits[i]['proxy'] = flattened_logit_proxyset

    logging.info(f'[Generating logits testset]')
    logit_testset = []
    with torch.no_grad():
        for elems, labels in test_dataloader:
            elems = elems.to(device)
            outputs = [model(elems).detach().cpu() for model in trained_models]
            stacked_outputs = torch.hstack(outputs)
            logit_testset.append((stacked_outputs, labels))
    flattened_logit_testset = []
    for x, y in logit_testset:
        flattened_logit_testset.extend(zip(x, y))
    logits['test'] = flattened_logit_testset    

    return logits

def generate_logits(trained_models, FedDataset, num_classes, proxy_frac, 
                    device, test_dataloader, num_clients, batch_size, seed):
    logits = {}

    trained_models = [model.to(device) for model in trained_models]

    for i in range(num_clients):
        logging.info(f'[Generating logits client {i}]')
        logits[i] = {}

        train_dataset, proxy_dataset, _ = load_trainset(i, FedDataset, num_classes, proxy_frac=proxy_frac, seed=seed)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        proxy_dataloader = torch.utils.data.DataLoader(
            proxy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        logit_trainset = []
        with torch.no_grad():
            for elems, labels in train_dataloader:
                elems = elems.to(device)
                outputs = [model(elems).detach().cpu() for model in trained_models]
                stacked_outputs = torch.hstack(outputs)
                logit_trainset.append((stacked_outputs, labels))
        flattened_logit_trainset = []
        for x, y in logit_trainset:
            flattened_logit_trainset.extend(zip(x, y))
        logits[i]['train'] = flattened_logit_trainset

        logit_proxyset = []
        with torch.no_grad():
            for elems, labels in proxy_dataloader:
                elems = elems.to(device)
                outputs = [model(elems).detach().cpu() for model in trained_models]
                stacked_outputs = torch.hstack(outputs)
                logit_proxyset.append((stacked_outputs, labels))
        flattened_logit_proxyset = []
        for x, y in logit_proxyset:
            flattened_logit_proxyset.extend(zip(x, y))
        logits[i]['proxy'] = flattened_logit_proxyset

    logging.info(f'[Generating logits testset]')
    logit_testset = []
    with torch.no_grad():
        for elems, labels in test_dataloader:
            elems = elems.to(device)
            outputs = [model(elems).detach().cpu() for model in trained_models]
            stacked_outputs = torch.hstack(outputs)
            logit_testset.append((stacked_outputs, labels))
    flattened_logit_testset = []
    for x, y in logit_testset:
        flattened_logit_testset.extend(zip(x, y))
    logits['test'] = flattened_logit_testset    

    return logits

def generate_logits_fedcam16(trained_models, FedDataset, num_classes, proxy_frac, 
                    device, test_dataloader, num_clients, batch_size, seed, collate_fn):
    logits = {}

    trained_models = [model.to(device) for model in trained_models]

    for i in range(num_clients):
        logging.info(f'[Generating logits client {i}]')
        logits[i] = {}

        train_dataset, proxy_dataset, _ = load_trainset_fedcam16(i, FedDataset, num_classes, proxy_frac=proxy_frac, seed=seed)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        proxy_dataloader = torch.utils.data.DataLoader(
            proxy_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        
        logit_trainset = []
        with torch.no_grad():
            for elems, labels in train_dataloader:
                elems = elems.to(device)
                outputs = [model(elems).detach().cpu() for model in trained_models]
                stacked_outputs = torch.hstack(outputs)
                logit_trainset.append((stacked_outputs, labels))
        flattened_logit_trainset = []
        for x, y in logit_trainset:
            flattened_logit_trainset.extend(zip(x, y))
        logits[i]['train'] = flattened_logit_trainset

        logit_proxyset = []
        with torch.no_grad():
            for elems, labels in proxy_dataloader:
                elems = elems.to(device)
                outputs = [model(elems).detach().cpu() for model in trained_models]
                stacked_outputs = torch.hstack(outputs)
                logit_proxyset.append((stacked_outputs, labels))
        flattened_logit_proxyset = []
        for x, y in logit_proxyset:
            flattened_logit_proxyset.extend(zip(x, y))
        logits[i]['proxy'] = flattened_logit_proxyset

    logging.info(f'[Generating logits testset]')
    logit_testset = []
    with torch.no_grad():
        for elems, labels in test_dataloader:
            elems = elems.to(device)
            outputs = [model(elems).detach().cpu() for model in trained_models]
            stacked_outputs = torch.hstack(outputs)
            logit_testset.append((stacked_outputs, labels))
    flattened_logit_testset = []
    for x, y in logit_testset:
        flattened_logit_testset.extend(zip(x, y))
    logits['test'] = flattened_logit_testset    

    return logits

# Function borrowed from: https://github.com/owkin/FLamby/blob/main/flamby/utils.py
def evaluate_model_on_tests(
    model, test_dataloaders, metric, use_gpu=True, return_pred=False, gpu_device=None
):
    """This function takes a pytorch model and evaluate it on a list of\
    dataloaders using the provided metric function.
    Parameters
    ----------
    model: torch.nn.Module,
        A trained model that can forward the test_dataloaders outputs
    test_dataloaders: List[torch.utils.data.DataLoader]
        A list of torch dataloaders
    metric: callable,
        A function with the following signature:\
            (y_true: np.ndarray, y_pred: np.ndarray) -> scalar
    use_gpu: bool, optional,
        Whether or not to perform computations on GPU if available. \
        Defaults to True.
    gpu_device: torch.device(), optional
        Which device to use if GPU is available. \
        Uses default cuda device if unspecified: model.cuda().
    Returns
    -------
    dict
        A dictionnary with keys client_test_{0} to \
        client_test_{len(test_dataloaders) - 1} and associated scalar metrics \
        as leaves.
    """
    results_dict = {}
    y_true_dict = {}
    y_pred_dict = {}
    if torch.cuda.is_available() and use_gpu:
        if gpu_device:
            model = model.to(gpu_device)
        else:
            model = model.cuda()
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            y_pred_final = []
            y_true_final = []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.to(gpu_device) if gpu_device else X.cuda()
                    y = y.to(gpu_device) if gpu_device else y.cuda()
                y_pred = model(X).detach().cpu()
                y = y.detach().cpu()
                y_pred_final.append(y_pred.numpy())
                y_true_final.append(y.numpy())

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            results_dict[f"client_test_{i}"] = metric(y_true_final, y_pred_final)
            if return_pred:
                y_true_dict[f"client_test_{i}"] = y_true_final
                y_pred_dict[f"client_test_{i}"] = y_pred_final
    
    # Put the model back in CPU memory to release GPU space
    cpu_device = torch.device('cpu')
    model.to(cpu_device)

    if return_pred:
        return results_dict, y_true_dict, y_pred_dict
    else:
        return results_dict
    
def get_weighting_matrix(my_lds):
    label_distributions = torch.tensor(my_lds)
    per_class_total = torch.sum(label_distributions, dim=0)
    weighting_matrix = torch.div(label_distributions, per_class_total)
    return weighting_matrix

def get_weighted_average(inferences, weighting_matrix):
    inferences = torch.tensor(inferences, dtype=torch.double)
    weighted_outputs = torch.mul(weighting_matrix, inferences)
    weighted_average = torch.sum(weighted_outputs, dim=0)
    return list(weighted_average)

def process_batch_inferences(batch_inferences, weighting_matrix):
    inferences_per_elem = dict()
    for m_i in range(len(batch_inferences)):
        batch_inference = batch_inferences[m_i]

        if(len(inferences_per_elem) == 0):
            for j in range(len(batch_inference)):
                inferences_per_elem[j] = [list(batch_inference[j])]
        else:
            for j in range(len(batch_inference)):
                inferences_per_elem[j].append(list(batch_inference[j]))
    
    averaged_inferences = []
    for k in range(len(inferences_per_elem)):
        weighted_average = get_weighted_average(inferences_per_elem[k], weighting_matrix)
        averaged_inferences.append(weighted_average)
    
    return torch.tensor(averaged_inferences)