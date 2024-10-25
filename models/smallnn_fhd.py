from torch import nn
import torch.nn.functional as F

class SmallNN_FHD_v3(nn.Module):

    def __init__(self, d=6, total_clients=4, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(total_clients*num_classes, total_clients*d)
        self.fc2 = nn.Linear(total_clients*d, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

class SmallNN_FHD_v2(nn.Module):
    def __init__(self):
        super().__init__()
        d = 4
        total_clients = 4
        self.fc1 = nn.Linear(total_clients, total_clients*d)
        self.fc2 = nn.Linear(total_clients*d, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x
    
class SmallNN_FHD(nn.Module):
    def __init__(self):
        super().__init__()
        total_clients = 4
        self.fc1 = nn.Linear(total_clients, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x