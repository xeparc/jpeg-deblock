from typing import *
import torch
import torch.nn as nn
import torchvision
from torchvision.ops import Conv2dNormActivation



class MobileNetQA(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        mobi = torchvision.models.mobilenet_v2()
        if in_channels != 3:
            self.stem = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.stem = nn.Identity()
        self.features = mobi.features
        self.head = nn.LazyLinear(1)

    def forward(self, x):
        x = self.stem(x)
        features = self.features(x)
        vec = features.mean(dim=(2,3))
        return 100 * (0.5 + self.head(vec))


class MobileNetIR(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, inverted_residual_setting=None) -> None:
        """
        MobileNet IR main class

        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()

        norm_layer = nn.BatchNorm2d

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n
                [4, 16, 1],
                [4, 24, 2],
                [4, 32, 3],
                [4, 64, 4],
            ]

        # building first layer
        ch = 16
        features = [
            Conv2dNormActivation(in_channels, ch, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]

        # building inverted residual blocks
        for t, c, n in inverted_residual_setting:
            for _ in range(n):
                features.append(InvertedResidual(ch, c, kernel_size=3, stride=1, expand_ratio=t))
                ch = c

        # building last several layers
        features.append(
            Conv2dNormActivation(
                ch, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class InvertedResidual(nn.Module):

    def __init__(self, inp: int, oup: int, kernel_size: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size-1) // 2,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
