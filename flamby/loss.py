import torch
from torch.nn.modules.loss import _Loss

class BaselineLoss_FHD(_Loss):
    def __init__(self, reduction="mean"):
        super(BaselineLoss_FHD, self).__init__(reduction=reduction)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target = target.float()
        return self.bce(input, target)