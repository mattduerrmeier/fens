import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline_FHD(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Baseline_FHD, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class SmallNN_FISIC(nn.Module):
    def __init__(self):
        super().__init__()
        d = 4
        total_clients = 6
        num_classes = 8
        self.fc1 = nn.Linear(total_clients*num_classes, total_clients*d)
        self.fc2 = nn.Linear(total_clients*d, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class SmallNN_FHD(nn.Module):
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

class SmallNN_FCAM(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        d = 4
        total_clients = 2
        self.fc1 = nn.Linear(total_clients, total_clients*d)
        self.fc2 = nn.Linear(total_clients*d, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x