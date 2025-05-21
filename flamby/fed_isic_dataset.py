import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from flamby.datasets.fed_isic2019 import FedIsic2019


class FedIsicCustom(Dataset):
    def __init__(self, train: bool, pooled: bool):
        # super().__init__()
        self.data = FedIsic2019(train=train, pooled=pooled)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x, y = self.data[idx]
        x = x.flatten(start_dim=0)
        y = F.one_hot(y, num_classes=8)
        return x, y
