from typing import *

import torch
import torch.nn as nn
import torchvision


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


class PerceptualLoss(nn.Module):

    def __init__(self,
                 layers: nn.Sequential,
                 indices: Iterable[int],
                 weights: Iterable[float],
                 norm: str = "l2",
                 input_transform: Optional[nn.Module] = None
    ):
        super().__init__()
        assert isinstance(layers, nn.Sequential)

        # Disable gradients
        self.layers = layers
        for param in self.layers.parameters():
            param.requires_grad = False

        self.indices = set(indices)
        self.weights = list(weights)
        self.max_idx = max(self.indices)
        assert len(self.indices) == len(self.weights)

        if norm == "l2":
            self.criterion = nn.MSELoss()
        elif norm == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError(norm)

        if input_transform is None:
            self.input_transform = nn.Identity()
        else:
            self.input_transform = input_transform

    def forward(self, x, target):
        x       = self.input_transform(x)
        target  = self.input_transform(target)

        x_features = []
        for i in range(self.max_idx):
            x = self.layers[i](x)
            if i in self.indices:
                x_features.append(x)

        with torch.no_grad():
            target_features = []
            for i in range(self.max_idx):
                target = self.layers[i](target)
                if i in self.indices:
                    target_features.append(target)

        loss = 0.0
        for xf, tf, w in zip(x_features, target_features, self.weights):
            loss += w * self.criterion(xf, tf)

        return loss