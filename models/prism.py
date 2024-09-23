import torch
import torch.nn as nn

from .blocks import ResidualDenseBlock4C


class PrismBackbone(nn.Module):
    """
    Luminance reconstruction net with 4 stages, 3 downsample layers.
    Downsampling factor = 2 after each stage.
    """

    def __init__(self, in_channels=1, out_channels=64, base_channels=16,
                 blocks_per_stage=1, channel_multiplier=2, use_3d_conv=False):
        super().__init__()
        self.use_3d_conv = use_3d_conv

        # Initialize stem
        ch = base_channels
        if use_3d_conv:
            self.stem = nn.Conv3d(in_channels, ch, kernel_size=(1,7,7), stride=1,
                                  padding=(0,3,3))
        else:
            self.stem = nn.Conv2d(in_channels, ch, kernel_size=7, stride=1,
                                  padding=3)

        # Initialize residual dense blocks
        self.stages = nn.ModuleList()
        for _ in range(4):
            gc = ch // 2
            stage = nn.Sequential(
                *[ResidualDenseBlock4C(ch, growth=gc, weight_scale=0.1)
                    for _ in range(blocks_per_stage)]
            )
            self.stages.append(stage)
            ch = int(ch * channel_multiplier)

        # Initialize downscale layers
        self.downscales = nn.ModuleList()
        ch = base_channels
        for _ in range(3):
            outch = int(channel_multiplier * ch)
            self.downscales.append(
                nn.Conv2d(ch, outch, kernel_size=2, stride=2)
            )
            ch = outch

        # Initialize head
        self.head = nn.Conv2d(outch, out_channels, kernel_size=1)

        # Initialize weights
        init_kwargs = dict(nonlinearity="leaky_relu", a=0.2)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(layer.weight, **init_kwargs)
                layer.weight.data *= 0.1

    def forward(self, x):
        if self.use_3d_conv:
            x = x.unsqueeze(2) # (N, Cin,  1, H, W)
            out = self.stem(x) # (N, Cout, 1, H, W)
            out = out.squeeze(2)
        else:
            out = self.stem(x)

        for i in range(3):
            out = self.stages[i](out)
            out = self.downscales[i](out)
        out = self.stages[3](out)
        out = self.head(out)
        return out


class PrismLumaS4(nn.Module):

    def __init__(self, idct, residual=False, base_channels=16,
                 blocks_per_stage=1, channel_multiplier=2):
        super().__init__()
        self.residual = residual
        self.luma = PrismBackbone(1, 64, base_channels, blocks_per_stage,
                                  channel_multiplier, False)
        self.idct = idct

    def forward(self, y, dct_y):
        # Extract Y plane
        r = self.luma(y)
        dct = r + dct_y if self.residual else r
        out = self.idct(dct, chroma=False)
        return out


class PrismNetS4(nn.Module):

    def __init__(self):
        super().__init__()
        self.luma = PrismBackbone(1, 64, base_channels=16, use_3d_conv=False)
        self.chroma = PrismBackbone(1, 128, base_channels=16, use_3d_conv=True)
        self.idct = 0

    def forward(self, ycc, dctY, dctCb, dctCr):
        # Extract Y, Cb, Cr planes
        Y, CbCr = ycc[:, :1, :, :], ycc[:, 1:, :, :]
        dctY_residium = self.luma(Y)
        # TODO