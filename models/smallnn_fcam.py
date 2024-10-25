from torch import nn
import torch.nn.functional as F

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
    
class SmallNN_FCAM_v2(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        total_clients = 2
        self.fc1 = nn.Linear(total_clients, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x