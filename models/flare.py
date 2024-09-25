import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional

from .blocks import (
    ConvertYccToRGB,
    FiLM,
    InverseDCT,
    ResidualDenseBlock4C,
)


class FlareBackbone(nn.Module):

    def __init__(self, in_channels, out_channels, base_channels=32,
                 blocks_per_stage=2, channel_multiplier=2, weight_scale=0.1):

        super().__init__()

        # Stem
        ch = base_channels
        # if use_3d_conv:
        #     self.stem = nn.Conv3d(in_channels, ch, (1,7,7), 1, (0,3,3))
        # else:
        #     self.stem = nn.Conv2d(in_channels, ch, 7, 1, 3)
        self.stem = nn.Conv2d(in_channels, base_channels, 7, 1, 3)

        # Residual dense stages
        self.stages = nn.ModuleList()
        stage_channels = []
        for _ in range(4):
            stage_channels.append(ch)
            gc = ch // 2
            stage = nn.Sequential(
                *[ResidualDenseBlock4C(ch, growth=gc, weight_scale=weight_scale)
                    for _ in range(blocks_per_stage)]
            )
            self.stages.append(stage)
            ch = int(ch * channel_multiplier)

        # FiLM (Feature-wise Linear Modulation) layers
        self.films = nn.ModuleList([
            FiLM(stage_channels[0], 64),
            FiLM(stage_channels[1], 64),
            FiLM(stage_channels[2], 64),
            FiLM(stage_channels[3], 64)
        ])

        # Downscaling layers
        self.downscales = nn.ModuleList([
            nn.Conv2d(stage_channels[0], stage_channels[1], 2, 2),
            nn.Conv2d(stage_channels[1], stage_channels[2], 2, 2),
            nn.Conv2d(stage_channels[2], stage_channels[3], 2, 2),
        ])

        # Head
        self.head = nn.Conv2d(stage_channels[3], out_channels, 1)

        # Init weights
        init_kwargs = dict(nonlinearity="leaky_relu", a=0.2)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(layer.weight, **init_kwargs)
                layer.weight.data *= weight_scale

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        # if self.use_3d_conv:
        #     x = x.unsqueeze(1)      # (N, 1, Cin, H, W)
        #     out = self.stem(x)      # (N, Cout, 3, H, W)
        #     out = out.squeeze(2)    # (N, Cout, H, W)
        # else:
        #     out = self.stem(x)
        out = self.stem(x)

        for i in range(3):
            out = self.stages[i](out)
            out = self.films[i](out, condition)
            out = self.downscales[i](out)
        out = self.stages[3](out)
        out = self.films[3](out, condition)

        return self.head(out)


class FlareLuma(nn.Module):

    def __init__(self, idct: InverseDCT, residual=True, base_channels=32,
                 blocks_per_stage=1, channel_multiplier=2):
        super().__init__()
        self.residual   = residual
        self.idct       = idct
        self.luma       = FlareBackbone(1, 64, base_channels, blocks_per_stage,
                                        channel_multiplier)

    def forward(self, y, dct_y, qt):
        r = self.luma(y, qt)
        dct = r + dct_y if self.residual else r
        out = self.idct(dct, chroma=False)
        return out


class FlareChroma(nn.Module):

    def __init__(self, idct: InverseDCT, residual=False, base_channels=32,
                 blocks_per_stage=1, channel_multiplier=2):
        super().__init__()
        self.residual   = residual
        self.idct       = idct
        self.chroma     = FlareBackbone(3, 128, base_channels, blocks_per_stage,
                                        channel_multiplier)

    def forward(self, y, cb, cr, dct_cb, dct_cr, qt_c):
        # Optionally downscale `y` tensor if chroma subsampling != 444
        ysize = y.shape[-2:]
        csize = cb.shape[-2:]
        if ysize != csize:
            y = torchvision.transforms.functional.resize(y, csize)
        # Prepare input
        x = torch.cat([y, cb, cr], dim=1)
        # Compute DCT coefficients (or residuals) for Cb, Cr
        cbcr = self.chroma(x, qt_c)
        cb_r, cr_r = torch.split(cbcr, 64, dim=1)
        dct_cb = cb_r + dct_cb if self.residual else cb_r
        dct_cr = cr_r + dct_cr if self.residual else cr_r
        # Apply inverse DCT transform to get Cb, Cr channels
        cb = self.idct(dct_cb, chroma=True)
        cr = self.idct(dct_cr, chroma=True)
        return cb, cr


class FlareNet(nn.Module):

    def __init__(self, luma_net: nn.Module, chroma_net: nn.Module, rgb_output=True,
                 luma_params="", freeze_luma=False):
        super().__init__()
        self.rgb_output     = rgb_output
        self.luma           = luma_net
        self.chroma         = chroma_net
        self.out_transform  = ConvertYccToRGB() if rgb_output else nn.Identity()
        if luma_params:
            state = torch.load(luma_params, weights_only=True, map_location="cpu")
            self.luma.load_state_dict(state)
        if freeze_luma:
            for param in self.luma.parameters():
                param.requires_grad = False

    def forward(self, y, cb, cr, dct_y, dct_cb, dct_cr, qt_y, qt_c):
        y_r         = self.luma(y, dct_y, qt_y)
        cb_r, cr_r  = self.chroma(y_r, cb, cr, dct_cb, dct_cr, qt_c)
        # Optionally upsample chroma
        ysize = y_r.shape[-2:]
        csize = cb_r.shape[-2:]
        if ysize != csize:
            cb_r = torchvision.transforms.functional.resize(cb_r, ysize)
            cr_r = torchvision.transforms.functional.resize(cr_r, ysize)
        ycc = torch.cat([y_r, cb_r, cr_r], dim=1)
        return self.out_transform(ycc)
