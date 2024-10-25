import sys
sys.path.append('../')

from .Dataset import Dataset

import torch
import torch.utils.data.distributed
from torch.utils.data import random_split
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
import logging
import os
import numpy as np
from sklearn import metrics

class FedISIC2019Agg(Dataset):

    def __init__(self, size, args):
        super().__init__(size, args)

        self.trainset = None
        self.proxyset = None
        self.valset = None
        self.testset = None
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.generator = torch.Generator().manual_seed(args.seed)
        
        if not self.args.logitpath:
            raise ValueError('Logit path must be specified for FedISIC2019Agg')

        self.generator = torch.Generator().manual_seed(self.args.seed)
        self.num_classes = 8
        self.load_trainset()
        self.load_testset()

    def load_trainset(self):
        logging.info('==> load train data')
        
        logit_trainset_path = os.path.join(self.args.logitpath, 'logits.pth')
        self.trainset = torch.load(logit_trainset_path)

        # Set num_samples
        new_num_samples = []
        for i in range(self.size):
            client_data, proxy_data = self.trainset[i]['train'], self.trainset[i]['proxy']
            data_len = len(proxy_data)
            if self.args.include_trainset:
                data_len += int(len(client_data) * self.args.include_trainset_frac)
            new_num_samples.append(data_len)
        self.num_samples = np.array(new_num_samples)
    
    def load_testset(self):
        logging.info('==> load test data')

        logit_testset_path = os.path.join(self.args.logitpath, 'logits.pth')
        logit_testset = torch.load(logit_testset_path)['test']
        
        if self.args.val_set:
            val_len = int(len(logit_testset) * self.args.val_ratio)
            self.valset, self.testset = random_split(logit_testset, [val_len, len(logit_testset) - val_len], generator=self.generator)
        else:
            self.testset = logit_testset
        
        logging.info('==> val set size: {}'.format(len(self.valset) if self.valset else 0))
        logging.info('==> test set size: {}'.format(len(self.testset)))

    def fetch(self, client_index):

        client_data, proxy_data = self.trainset[client_index]['train'], self.trainset[client_index]['proxy']
        
        train_loader = None
        if self.args.include_trainset:
            # Create a new rng every time to replicate the same subset of data
            tmp_rng = np.random.default_rng(self.args.seed)

            indices = list(range(len(client_data)))
            data_len = len(indices)
            sub_len = int(data_len * self.args.include_trainset_frac)
            indices = list(tmp_rng.choice(indices, sub_len, replace=False))
            client_data = [client_data[i] for i in indices]

            train_loader = torch.utils.data.DataLoader(client_data + proxy_data, 
                                            batch_size=self.args.bs, 
                                            shuffle=True, 
                                            pin_memory=True)
            
            logging.info('==> Client id {}, samples from trainset {}, samples from proxyset: {}' \
                        .format(client_index, len(client_data), len(proxy_data)))
    
        else:
            train_loader = torch.utils.data.DataLoader(proxy_data, 
                                        batch_size=self.args.bs, 
                                        shuffle=True, 
                                        num_workers=1)
        
            logging.info('==> Client id {}, samples from trainset {}, samples from proxyset: {}' \
                        .format(client_index, 0, len(proxy_data)))
    
        test_loader = torch.utils.data.DataLoader(self.testset, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=False, 
                                            num_workers=1)
        val_loader = None
        if self.valset:
            val_loader = torch.utils.data.DataLoader(self.valset, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=False, 
                                            num_workers=1)
        
        # TODO: if needed
        local_test_loader = None
        return train_loader, None, val_loader, test_loader, local_test_loader, self.num_samples
    
    # https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/metric.py
    @staticmethod
    def metric(y_true, preds):
        y_true = y_true.reshape(-1)
        return metrics.balanced_accuracy_score(y_true, preds)
    
    # https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/loss.py
    class criterion(_Loss):
        """Weighted focal loss
        See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
        a good explanation
        Attributes
        ----------
        alpha: torch.tensor of size 8, class weights
        gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
        the same as CE loss, increases gamma reduces the loss for the "hard to classify
        examples"
        """

        def __init__(
            self,
            alpha=torch.tensor(
                [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
            ),
            gamma=2.0,
        ):
            super().__init__()
            self.alpha = alpha.to(torch.float)
            self.gamma = gamma

        def forward(self, inputs, targets):
            """Weighted focal loss function
            Parameters
            ----------
            inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
            targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
            """
            targets = targets.view(-1, 1).type_as(inputs)
            logpt = F.log_softmax(inputs, dim=1)
            logpt = logpt.gather(1, targets.long())
            logpt = logpt.view(-1)
            pt = logpt.exp()
            self.alpha = self.alpha.to(targets.device)
            at = self.alpha.gather(0, targets.data.view(-1).long())
            logpt = logpt * at
            loss = -1 * (1 - pt) ** self.gamma * logpt

            return loss.mean()

