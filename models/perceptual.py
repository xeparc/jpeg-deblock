from typing import *

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

class PixelwiseFeatures(torch.nn.Module):
    """
    Implementation of

    "Pixelwise JPEG Compression Detection and Quality Factor Estimation
     Based on Convolutional Neural Network"

    paper in PyTorch. The last layer is missing, because this module is
    intended to be used as a feature extractor for perceptual loss, hence
    it's not needed.

    https://github.com/kuchida/PixelwiseJPEGCompressionDetection/
    """

    TRAINED_WEIGHTS = "../data/params/pixelwise/statedict.pth"

    def __init__(self, pretrained=False):
        super().__init__()
        kwargs = dict(kernel_size=3, padding="same")
        self.conv2d_1 = torch.nn.Conv2d( 3, 64, dilation=1, **kwargs)
        self.conv2d_2 = torch.nn.Conv2d(64, 64, dilation=2, **kwargs)
        self.conv2d_3 = torch.nn.Conv2d(64, 64, dilation=3, **kwargs)
        self.conv2d_4 = torch.nn.Conv2d(64, 64, dilation=4, **kwargs)
        self.conv2d_5 = torch.nn.Conv2d(64, 64, dilation=3, **kwargs)
        self.conv2d_6 = torch.nn.Conv2d(64, 64, dilation=2, **kwargs)
        self.conv_layers = [self.conv2d_1, self.conv2d_2, self.conv2d_3,
                            self.conv2d_4, self.conv2d_5, self.conv2d_6]
        self.num_layers = 6

        if pretrained:
            state_dict = torch.load(PixelwiseFeatures.TRAINED_WEIGHTS,
                                    weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            x = torch.nn.functional.relu(x)
        return x


class Q1Net(nn.Module):

        def __init__(self, in_channels=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, bias=False),
                Q1Bottleneck(32, multiplier=2),
                nn.Conv2d(32, 32, kernel_size=3, bias=False),
                Q1Bottleneck(32, multiplier=2),
                Conv2dNormActivation(32, 64, kernel_size=3, bias=False),
                Q1Bottleneck(64, multiplier=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv2dNormActivation(64, 128, kernel_size=3, bias=False),
                Q1Bottleneck(128, multiplier=2),
                Conv2dNormActivation(128, 128, kernel_size=3, bias=False),
                Q1Bottleneck(128, multiplier=2),
                Conv2dNormActivation(128, 128, kernel_size=3),
            )
            self.head = nn.Linear(128, 1)

        def forward(self, x):
            x = self.features(x)
            x = torch.mean(x, dim=(-2, -1))
            y = self.head(x)
            return 0.5 + y


class Q1Bottleneck(nn.Module):

    def __init__(self, channels, multiplier=2, kernel_size=3):
        super().__init__()
        mid = channels * multiplier
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            Conv2dNormActivation(mid, mid, kernel_size=kernel_size, padding="same"),
            nn.Conv2d(mid, channels, kernel_size=1)
        )

    def forward(self, x):
        y = self.bottleneck(x)
        return x + y

