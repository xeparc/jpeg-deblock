"""
Implementation of DnCNN model for JPEG artifact reduction.
---
From paper:
"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"

"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):

    def __init__(self, in_channels: int = 3, depth: int = 3):
        super().__init__()

        layers = []
        # Stem
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding="same"))
        layers.append(nn.ReLU())
        # Body
        for _ in range(depth):
            layers.append(nn.Conv2d(64, 64, 3, padding="same"))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU())
        # Head
        layers.append(nn.Conv2d(64, in_channels, kernel_size=3, padding="same"))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.layers(x)
        return x + residual


class DnCNN3(DnCNN):

    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels, 17)

