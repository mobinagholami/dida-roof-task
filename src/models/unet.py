import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.dec1(x)
        return x
