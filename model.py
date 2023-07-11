import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)
