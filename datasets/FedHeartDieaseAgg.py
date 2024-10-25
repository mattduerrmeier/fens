import sys
sys.path.append('../')

from .Dataset import Dataset

import torch
import torch.utils.data.distributed
from torch.utils.data import random_split
from torch.nn.modules.loss import _Loss
import logging
import os
import numpy as np

class FedHeartDiseaseAgg(Dataset):

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
            raise ValueError('Logit path must be specified for FedHeartDiseaseAgg')

        self.generator = torch.Generator().manual_seed(self.args.seed)
        self.num_classes = 2
        self.require_argmax = False
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
        
        # TODO: if required
        local_test_loader = None
        return train_loader, None, val_loader, test_loader, local_test_loader, self.num_samples
    
    # https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_heart_disease/metric.py
    def metric(self, y_true, y_pred):
        y_true = y_true.astype("uint8")
        # The try except is needed because when the metric is batched some batches
        # have one class only
        try:
            return ((y_pred > 0.0) == y_true).mean()
        except ValueError:
            return np.nan

    # https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_heart_disease/loss.py
    class criterion(_Loss):
        def __init__(self, reduction="mean"):
            super().__init__(reduction=reduction)
            self.bce = torch.nn.BCEWithLogitsLoss()

        def forward(self, input: torch.Tensor, target: torch.Tensor):
            target = target.reshape(input.shape).float()
            return self.bce(input, target)

