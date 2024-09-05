import torch
import torch.nn as nn


class SRCNN(nn.Module):

    def __init__(self,):
        super().__init__()
        # (33, 33, 3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=9, stride=1, padding="same")
        # (33, 33, 64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32,
                               kernel_size=1, stride=1, padding=0)
        # (33, 33, 32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3,
                               kernel_size=5, stride=1, padding="same")
        # (33, 33, 3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        h = self.relu1(self.conv1(x))
        g = self.relu2(self.conv2(h))
        y = torch.clip(self.conv3(g), 0.0, 1.0)
        return y

