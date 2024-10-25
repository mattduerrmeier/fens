import sys
sys.path.append('../')

from .Dataset import Dataset
from .DataPartitioner import DataPartitioner

import torch
import torch.utils.data.distributed
from torch.utils.data import random_split
import logging
import os
import numpy as np

class SVHNAgg(Dataset):

    def __init__(self, size, args):
        super().__init__(size, args)

        self.trainset = None
        self.proxyset = None
        self.valset = None
        self.testset = None
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.generator = torch.Generator().manual_seed(args.seed)
        
        if not self.args.proxy_set:
            raise ValueError('Proxy set must be specified for SVHN')
        if not self.args.logitpath:
            raise ValueError('Logit path must be specified for SVHNAgg')

        self.generator = torch.Generator().manual_seed(self.args.seed)
        self.num_classes = 10
        self.load_trainset()
        self.load_testset()

    def load_trainset(self):
        logging.info('==> load train data')
        
        logit_trainset_path = os.path.join(self.args.logitpath, 'logit_trainset.pth')
        logit_trainset = torch.load(logit_trainset_path)

        # Take a subset of the combined dataset if specified
        if self.args.tr_subset:
            data_len = len(logit_trainset)
            sub_len = int(data_len * self.args.tr_subset_frac)
            logit_trainset, _ = random_split(logit_trainset, [sub_len, data_len - sub_len], generator=self.generator)
        self.logit_trainset = logit_trainset

        partition_sizes = [1.0 / self.size for _ in range(self.size)]
        self.partition = DataPartitioner(self.logit_trainset, partition_sizes, isNonIID=self.args.NIID, num_classes=self.num_classes, \
                                        seed=self.args.seed, alpha=self.args.alpha, dataset=self.args.dataset, \
                                        proxyset=self.args.proxy_set, proxy_ratio=self.args.proxy_ratio)
        self.num_samples = self.partition.ratio # returns counts intead of ratios

        # Update num_samples based on the proxy ratio and include_trainset_frac
        new_num_samples = []
        for i in range(self.size):
            client_data, proxy_data = self.partition.use(i)
            data_len = len(proxy_data)
            if self.args.include_trainset:
                data_len += int(len(client_data) * self.args.include_trainset_frac)
            new_num_samples.append(data_len)
        self.num_samples = np.array(new_num_samples)
    
    def load_testset(self):
        logging.info('==> load test data')

        logit_testset_path = os.path.join(self.args.logitpath, 'logit_testset.pth')
        logit_testset = torch.load(logit_testset_path)
        
        if self.args.val_set:
            val_len = int(len(logit_testset) * self.args.val_ratio)
            self.valset, self.testset = random_split(logit_testset, [val_len, len(logit_testset) - val_len], generator=self.generator)
        else:
            self.testset = logit_testset
        
        logging.info('==> val set size: {}'.format(len(self.valset) if self.valset else 0))
        logging.info('==> test set size: {}'.format(len(self.testset)))

    def fetch(self, client_index):

        client_data, proxy_data = self.partition.use(client_index)
        
        train_loader = None
        if self.args.include_trainset:
            # Create a new rng every time to replicate the same subset of data
            tmp_rng = np.random.default_rng(self.args.seed)

            indices = client_data.index
            data_len = len(indices)
            sub_len = int(data_len * self.args.include_trainset_frac)
            indices = list(tmp_rng.choice(indices, sub_len, replace=False))
            client_data.update_index(indices)

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

