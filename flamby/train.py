import logging
import copy
import torch
import wandb

def train_and_evaluate(id_str, model, criterion, optimizer, scheduler, 
                        train_loader, test_loader, iterations, device, 
                        metric, test_every=5, require_argmax=False):

    wandb.define_metric(f'{id_str}/epoch')
    wandb.define_metric(f'{id_str}/*', step_metric=f'{id_str}/epoch')

    model.train()
    model = model.to(device)
    best_model = None

    losses = 0
    count = 0
    epoch = 0

    best_acc = 0.0

    y_preds = []
    y_trues = []
    
    while epoch < iterations:
        epoch += 1
        for data, target in train_loader:
            y_trues.append(target)

            # data loading for GPU
            data = data.to(device)
            target = target.to(device)

            # forward pass
            output = model(data)
            if not require_argmax:
                target = target.reshape(output.shape)
            loss = criterion(output, target)

            # backward pass
            loss.backward()

            # gradient step
            optimizer.step()
            optimizer.zero_grad()

            if require_argmax:
                y_preds.append(output.argmax(dim=1).detach().cpu())
            else:
                y_preds.append(output.detach().cpu())

            losses += loss.item() * data.size(0)
            count += data.size(0)

        scheduler.step()

        losses /= count
        y_preds_np = torch.cat(y_preds).numpy()
        y_preds_np = y_preds_np.squeeze(-1) if y_preds_np.shape[-1] == 1 else y_preds_np

        y_trues_np = torch.cat(y_trues).numpy()
        y_trues_np = y_trues_np.squeeze(-1) if y_trues_np.shape[-1] == 1 else y_trues_np

        acc = metric(y_trues_np, y_preds_np)    
        
        logging.info(f'Epoch {epoch} Train Loss {losses:.4f} Train Acc {acc:.4f}')
        wandb.log({f'{id_str}/train_loss': losses, f'{id_str}/train_acc': acc, 
                   f'{id_str}/epoch': epoch})

        if epoch % test_every == 0:
            test_loss, test_acc = evaluate(model, criterion, test_loader, device, 
                                           metric, require_argmax=require_argmax)
            if test_acc > best_acc:
                best_acc = test_acc
                # store the best model by making a copy
                best_model = copy.deepcopy(model)
            logging.info(f'Epoch {epoch} Test Loss {test_loss:.4f} Test Acc {test_acc:.4f}')
            wandb.log({f'{id_str}/test_loss': test_loss, f'{id_str}/test_acc': test_acc, 
                       f'{id_str}/epoch': epoch})

    return best_acc, best_model

def evaluate(model, criterion, test_loader, device, metric, require_argmax=True):
    model.eval()
    model.to(device)

    losses = 0
    counts = 0
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for data, target in test_loader:
            y_trues.append(target)

            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)
            if not require_argmax:
                target = target.reshape(outputs.shape)

            loss = criterion(outputs, target)
            losses += loss.item() * data.size(0)
            counts += data.size(0)
            
            if require_argmax:
                y_preds.append(torch.argmax(outputs, dim=1).detach().cpu())
            else:
                y_preds.append(outputs.detach().cpu())

    losses /= counts
                    
    y_preds_np = torch.cat(y_preds).numpy()
    y_preds_np = y_preds_np.squeeze(-1) if y_preds_np.shape[-1] == 1 else y_preds_np

    y_trues_np = torch.cat(y_trues).numpy()
    y_trues_np = y_trues_np.squeeze(-1) if y_trues_np.shape[-1] == 1 else y_trues_np

    acc = metric(y_trues_np, y_preds_np)
    
    return losses, acc