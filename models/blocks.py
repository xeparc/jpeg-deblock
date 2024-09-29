import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock4C(nn.Module):

    def __init__(self, in_channels=64, growth=32, weight_scale=0.1, bias=True):
        super().__init__()
        self.channels = in_channels
        self.negative_slope = 0.2
        self.weight_scale = weight_scale
        self.residual_scale = 0.2
        # growth: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(in_channels, growth, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels + 1 * growth, growth, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth, growth, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth, in_channels, 3, 1, 1, bias=bias)
        self.relu  = nn.LeakyReLU(self.negative_slope, inplace=True)
        # Initialization
        init_kwargs = dict(nonlinearity="leaky_relu", a=self.negative_slope)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, **init_kwargs)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, **init_kwargs)
        torch.nn.init.kaiming_uniform_(self.conv3.weight, **init_kwargs)
        torch.nn.init.kaiming_uniform_(self.conv4.weight, **init_kwargs)
        self.conv1.weight.data *= weight_scale
        self.conv2.weight.data *= weight_scale
        self.conv3.weight.data *= weight_scale

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return x4 * self.residual_scale + x

class ResidualDenseBlock4DSC(nn.Module):
    """Residual Dense Block with 4 depthwise separable convolutional layers."""

    def __init__(self, in_channels=64, growth=32, weight_scale=0.1, bias=True):
        super().__init__()
        self.channels = in_channels
        self.negative_slope = 0.2
        self.weight_scale = weight_scale
        self.residual_scale = 0.2
        # growth: growth channel, i.e. intermediate channels
        conv2d_kwargs = dict(kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv1 = DSConv2d(in_channels, growth, weight_scale, **conv2d_kwargs)
        self.conv2 = DSConv2d(in_channels + 1 * growth, growth, weight_scale, **conv2d_kwargs)
        self.conv3 = DSConv2d(in_channels + 2 * growth, growth, weight_scale, **conv2d_kwargs)
        self.conv4 = DSConv2d(in_channels + 3 * growth, in_channels, weight_scale, **conv2d_kwargs)
        self.relu  = nn.LeakyReLU(self.negative_slope, inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return x4 * self.residual_scale + x


class DSConv2d(nn.Module):
    """Depthwise Separable Conv2d"""

    def __init__(self, in_channels, out_channels, weight_scale=0.1, **conv2d_kwargs):
        super().__init__()
        if in_channels < out_channels:
            mid_channels = in_channels
        else:
            mid_channels = out_channels
        self.dsconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.Conv2d(mid_channels, mid_channels, groups=mid_channels, **conv2d_kwargs),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        )
        # Initialization
        torch.nn.init.xavier_uniform_(self.dsconv[0].weight)
        torch.nn.init.xavier_uniform_(self.dsconv[2].weight)
        torch.nn.init.kaiming_uniform_(self.dsconv[1].weight, nonlinearity="leaky_rely", a=0.2)
        self.dsconv[1].weight.data *= weight_scale

    def forward(self, x):
        return self.dsconv(x)


class RRDB4(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, in_channels, growth=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock4C(in_channels, growth)
        self.RDB2 = ResidualDenseBlock4C(in_channels, growth)
        self.RDB3 = ResidualDenseBlock4C(in_channels, growth)
        self.RDB4 = ResidualDenseBlock4C(in_channels, growth)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = self.RDB4(out)
        return out * 0.2 + x


class  ToDCTTensor:

    def __init__(self,
                 luma_mean=None, luma_std=None,
                 chroma_mean=None, chroma_std=None,
        ):
        if luma_mean is not None:
            assert luma_mean.shape == (8,8)
            self.luma_mean = torch.as_tensor(luma_mean).float()
        else:
            self.luma_mean = torch.zeros((8,8), dtype=torch.float32)

        if luma_std is not None:
            assert luma_std.shape == (8,8)
            self.luma_std = torch.as_tensor(luma_std).float()
        else:
            self.luma_std = torch.ones((8,8), dtype=torch.float32)

        if luma_mean is None and luma_std is None:
            self.skip_luma = True
        else:
            self.skip_luma = False

        if chroma_mean is not None:
            assert chroma_mean.shape == (8,8)
            self.chroma_mean = torch.as_tensor(chroma_mean).float()
        else:
            self.chroma_mean = torch.zeros((8,8), dtype=torch.float32)

        if chroma_std is not None:
            assert chroma_std.shape == (8,8)
            self.chroma_std = torch.as_tensor(chroma_std).float()
        else:
            self.chroma_std = torch.ones((8,8), dtype=torch.float32)

        if chroma_mean is None and chroma_std is None:
            self.skip_chroma = True
        else:
            self.skip_chroma = False

    def __call__(self, dct: np.ndarray, chroma: bool):
        assert dct.ndim == 4
        assert dct.shape[2] == dct.shape[3] == 8
        out_shape = dct.shape[:-2] + (64,)
        dct = torch.from_numpy(dct)
        if chroma:
            if self.skip_chroma:
                res = dct
            else:
                res = (dct - self.chroma_mean) / self.chroma_std
        else:
            if self.skip_luma:
                res = dct
            else:
                res = (dct - self.luma_mean) / self.luma_std
        return res.view(out_shape).permute(2,0,1).contiguous()


class ToQTTensor(nn.Module):

    def __init__(self, invert=False):
        self.invert = invert

    def __call__(self, qtable: np.ndarray):
        x = torch.as_tensor((qtable.astype(np.float32) - 1) / 254).ravel()
        if self.invert:
            return 1.0 - x
        else:
            return x


class ConvertYccToRGB(torch.nn.Module):

    def __init__(self):
        """Transforms image from YCbCr color space to RGB.
        Range of pixels is assumed to be [0,1]."""
        super().__init__()

        # Uses values from libjpeg-6b. See "jdcolor.c"
        #
        # * The conversion equations to be implemented are therefore
        # *	R = Y                + 1.40200 * Cr
        # *	G = Y - 0.34414 * Cb - 0.71414 * Cr
        # *	B = Y + 1.77200 * Cb
        # * where Cb and Cr represent the incoming values less CENTERJSAMPLE.
        matrix = torch.tensor(
            [[1.0,  0.0,       1.40200],
             [1.0,  -0.34414, -0.71414],
             [1.0,  1.772,         0.0]], dtype=torch.float32
        )
        offset = torch.tensor([0.0, -128/255, -128/255], dtype=torch.float32)
        self.register_buffer("conv_matrix", matrix)
        self.register_buffer("offset", offset)

    def forward(self, x):
        # x.shape =                 (B,3,H,W)
        # self.conv_matrix.shape  = (3,3)
        # Subtract 0.5 from CbCr
        yuv = x + self.offset.view(1, 3, 1, 1)
        rgb = torch.einsum("rc,cbhw->rbhw", self.conv_matrix, yuv.transpose(0,1))
        return rgb.transpose(0,1)


class ConvertRGBToYcc(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Uses values from libjpeg-6b. See "jccolor.c"
        #
        # * The conversion equations to be implemented are therefore
        # *	Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
        # *	Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + CENTERJSAMPLE
        # *	Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + CENTERJSAMPLE
        matrix = torch.tensor(
            [[ 0.29900,  0.58700,  0.11400],
             [-0.16874, -0.33126,  0.50000],
             [ 0.50000, -0.41869, -0.08131]], dtype=torch.float32
        )
        offset = torch.tensor([0.0, 128/255, 128/255], dtype=torch.float32)
        self.register_buffer("conv_matrix", matrix)
        self.register_buffer("offset", offset)

    def forward(self, x):
        yuv = torch.einsum("rc,cbhw->rbhw", self.conv_matrix, x.transpose(0,1))
        yuv = yuv.transpose(0,1) + self.offset.view(1, 3, 1, 1)
        return yuv


class InverseDCT(torch.nn.Module):

    def __init__(self, luma_mean=None, luma_std=None,
                       chroma_mean=None, chroma_std=None):
        super().__init__()

        # Make Type III harmonics
        steps = torch.arange(8, requires_grad=False) / 16
        f = 2 * torch.arange(8, requires_grad=False) + 1
        h1 = torch.cos(torch.outer(steps, f * torch.pi))
        h2 = h1.clone()
        # Make IDCT basis
        basis = h1.T.view(8, 1, 8, 1) * h2.T.view(1, 8, 1, 8)
        self.register_buffer("basis", basis)        # (8,8, 8,8)
        # Make normalization matrix
        c = torch.ones(8, dtype=torch.float32)
        c[0] = torch.sqrt(torch.tensor(0.5, dtype=torch.float32))
        C = 0.25 * torch.outer(c, c)
        self.register_buffer("scale", C)

        if luma_mean is None:
            self.register_buffer("luma_mean", torch.zeros(64).float())
        else:
            self.register_buffer("luma_mean", torch.as_tensor(luma_mean).float())

        if luma_std is None:
            self.register_buffer("luma_std", torch.ones(64).float())
        else:
            self.register_buffer("luma_std", torch.as_tensor(luma_std).float())

        if chroma_mean is None:
            self.register_buffer("chroma_mean", torch.zeros(64).float())
        else:
            self.register_buffer("chroma_mean", torch.as_tensor(chroma_mean).float())

        if chroma_std is None:
            self.register_buffer("chroma_std", torch.ones(64).float())
        else:
            self.register_buffer("chroma_std", torch.as_tensor(chroma_std).float())

    def forward(self, dct: torch.FloatTensor, chroma: bool):
        # (B, 64, H, W)
        B, C, H, W = dct.shape
        assert C == 64
        # Normalize
        if chroma:
            out = dct * self.chroma_std.view(1, 64, 1, 1) + self.chroma_mean.view(1, 64, 1, 1)
        else:
            out = dct * self.luma_std.view(1, 64, 1, 1) + self.luma_mean.view(1, 64, 1, 1)
        # Reshape
        out = out.view(B, 1, 1, 8, 8, H, W)
        C = self.scale.view(1, 1, 1, 8, 8, 1, 1)
        basis = self.basis.view(1, 8, 8, 8, 8, 1, 1)
        # Apply transform
        res = torch.sum(C * basis * out, dim=(3,4))         # (B, 8, 8, H, W)
        res = res.permute(0, 3, 1, 4, 2).contiguous()       # (B, H, 8, W, 8)

        return (res.view(B, 1, 8*H, 8*W) + 128) / 255.0


class FiLM(nn.Module):

    def __init__(self, features_dim: int, condition_dim: int):
        # TODO
        # Allow condition_dim to be `tuple`. Easy to implement
        super().__init__()

        self.f_gamma = nn.Sequential(
            nn.Linear(condition_dim, features_dim, bias=False))
        self.f_beta  = nn.Sequential(
            nn.Linear(condition_dim, features_dim, bias=False))

        nn.init.xavier_uniform_(self.f_gamma[0].weight)
        nn.init.xavier_uniform_(self.f_beta[0].weight, gain=0.1)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            features:
                Feature maps with shape (N, C, H, W)
            condition:
                Vector with shape (N, K)
        """
        N = condition.shape[0]
        gamma = self.f_gamma(condition).view(N, -1, 1, 1)
        beta  = self.f_beta(condition).view(N, -1, 1, 1)
        return gamma * features + beta


