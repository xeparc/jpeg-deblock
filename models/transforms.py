import numpy as np
import torch
import torch.nn as nn


class  ToDCTTensor:
    """Converts DCT coefficient matrix to Torch tensor"""

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
        dct = torch.as_tensor(dct)
        assert dct.ndim == 4
        assert dct.shape[2] == dct.shape[3] == 8
        out_shape = dct.shape[:-2] + (64,)
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
        qtable = torch.as_tensor(qtable).float()
        x = ((qtable - 1) / 254).ravel()
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
