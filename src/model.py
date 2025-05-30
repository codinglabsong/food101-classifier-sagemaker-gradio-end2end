import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """A tiny CNN for MNIST – ~12 k params."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 28×28 → 14×14
            nn.Conv2d(32, 64, 3, padding=1), # 14×14 → 14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14×14 → 7×7
        )
        self.fc   = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
