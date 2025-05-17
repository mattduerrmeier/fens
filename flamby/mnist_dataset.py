import torch
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
from typing import cast, List
import numpy as np
import numpy.random as npr

NUM_CENTERS = 2


def split(dataset: Dataset, nr_clients: int, iid: bool, seed: int) -> List[Subset]:
    rng = npr.default_rng(seed)

    if iid:
        # shuffle the data around and split per clients
        splits = np.array_split(rng.permutation(len(dataset)), nr_clients)
    else:
        # sort the dataset per class
        sorted_indices = np.argsort(np.array([target for _data, target in dataset]))
        # shard per class, times 2
        shards = np.array_split(sorted_indices, 2 * nr_clients)
        shuffled_shard_indices = rng.permutation(len(shards))
        splits = [
            np.concatenate([shards[i] for i in shard_ind_pairs], dtype=np.int64)
            for shard_ind_pairs in shuffled_shard_indices.reshape(nr_clients, 2)
        ]

    return [Subset(dataset, split) for split in cast(list[list[int]], splits)]


class MNISTDataset(Dataset):
    list_splits = None

    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        download: bool = True,
    ):
        assert center in [*range(0, NUM_CENTERS)]

        transforms = Compose(
            [
                Grayscale(),
                ToTensor(),
            ]
        )

        dataset = MNIST(
            root="data/",
            train=train,
            transform=transforms,
            download=download,
        )

        if pooled:
            self.data = dataset
        else:
            if MNISTDataset.list_splits is None:
                MNISTDataset.list_splits = split(
                    dataset, nr_clients=NUM_CENTERS, iid=True, seed=42
                )

            center_split = MNISTDataset.list_splits[center]
            self.data = center_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = x.flatten(start_dim=0)
        y = F.one_hot(torch.tensor(y), num_classes=10)
        return x, y
