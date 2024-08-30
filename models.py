import copy
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from jpegutils import SUBSAMPLE_FACTORS

DIM_QTABLE = 64

class ARCNN(nn.Module):

    # The first layer performs patch extraction and representation
    # Then the non-linear mapping layer maps each high-dimensional vector of
    # the first layer to another high-dimensional space
    # At last, thers is a reconstruction layer, which aggregates the patch-wise
    # representations to generate the final output.

    # There is no pooling or full-connected layers in SRCNN, so the final
    # output F(Y) is of the same size as the input image.

    # AR-CNN consists fo four layers:
    #   - feature extraction
    #   - feature enhancement
    #   - mapping
    #   - reconstruction layer

    # AR-CNN uses Parametric Rectified Linear Unit (PReLU):
    #   PReLU(x) = max(x, 0) + a * min(0,x)
    # PReLU is mainly used to avoid the "dead features" caused by zero
    # gradients in ReLU

    # AR-CNN is not equal to deeper SRCNN that contains more than one non-linear
    # mapping layers.

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               dtype=torch.float32,
                               kernel_size=9, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, dtype=torch.float32,
                               kernel_size=7, stride=1, padding="same")
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, dtype=torch.float32,
                               kernel_size=1, stride=1, padding="same")
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=3, dtype=torch.float32,
                               kernel_size=5, stride=1, padding="same")

        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)

        # Initialize
        with torch.no_grad():
            for layer in (self.conv1, self.conv2, self.conv3):
                nn.init.dirac_(layer.weight)
                dirac = torch.clone(layer.weight)
                nn.init.normal_(layer.weight, mean=0.0, std=1e-2)
                noise = torch.clone(layer.weight)
                layer.weight = nn.Parameter(noise + dirac)

            # nn.init.normal_(self.conv1.weight, std=1e-2)
            # nn.init.normal_(self.conv2.weight, std=1e-2)
            # nn.init.normal_(self.conv3.weight, std=1e-2)
            # nn.init.normal_(self.conv4.weight, std=1e-2)

    def forward(self, x):
        h0 = self.relu1(self.conv1(x))
        h1 = self.relu2(self.conv2(h0))
        h2 = self.relu3(self.conv3(h1))
        y  = self.conv4(h2)
        return y


class ResNetD(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               dtype=torch.float32,
                               kernel_size=9, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, dtype=torch.float32,
                               kernel_size=7, stride=1, padding="same")
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, dtype=torch.float32,
                               kernel_size=1, stride=1, padding="same")
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=3, dtype=torch.float32,
                               kernel_size=5, stride=1, padding="same")

        self.relu1 = nn.LeakyReLU(negative_slope=0.25)
        self.relu2 = nn.LeakyReLU(negative_slope=0.25)
        self.relu3 = nn.LeakyReLU(negative_slope=0.25)

        # Initialize

    def forward(self, x):
        h0 = self.relu1(self.conv1(x))
        h1 = self.relu2(self.conv2(h0))
        h2 = self.relu3(self.conv3(h1))
        y  = x + self.conv4(h2)
        return y


class ResNetD2(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128,
                               dtype=torch.float32,
                               kernel_size=9, stride=1, padding="same", padding_mode="reflect")
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, dtype=torch.float32,
                               kernel_size=7, stride=1, padding="same", padding_mode="reflect")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, dtype=torch.float32,
                               kernel_size=1, stride=1, padding="same", padding_mode="reflect")
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, dtype=torch.float32,
                               kernel_size=7, stride=1, padding="same", padding_mode="reflect")
        self.conv5 = nn.Conv2d(in_channels=3, out_channels=64, dtype=torch.float32,
                               kernel_size=7, stride=1, padding="same", padding_mode="reflect")
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=3, dtype=torch.float32,
                               kernel_size=7, stride=1, padding="same", padding_mode="reflect")

        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)

        # # Initializeß
        # with torch.no_grad():
        #     for layer in (self.conv5, self.conv6):
        #         nn.init.dirac_(layer.weight)
        #         dirac = torch.clone(layer.weight)
        #         nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        #         noise = torch.clone(layer.weight)
        #         layer.weight = nn.Parameter(noise + dirac)

    def forward(self, x):
        h0 = self.relu1(self.conv1(x))
        h1 = self.relu2(self.conv2(h0))
        h2 = self.relu3(self.conv3(h1))
        h3 = self.relu4(x + self.conv4(h2))
        h4 = self.relu5(self.conv5(h3))
        y = self.conv6(h4)
        return y


class ResNetD3(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128,
                               dtype=torch.float32,
                               kernel_size=9, stride=1, padding="same", padding_mode="reflect")
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, dtype=torch.float32,
                               kernel_size=7, stride=1, padding="same", padding_mode="reflect")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, dtype=torch.float32,
                               kernel_size=1, stride=1, padding="same", padding_mode="reflect")
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, dtype=torch.float32,
                               kernel_size=3, stride=1, padding="same", padding_mode="reflect")
        self.conv5 = nn.Conv2d(in_channels=3, out_channels=64, dtype=torch.float32,
                               kernel_size=3, stride=1, padding="same", padding_mode="reflect")
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=3, dtype=torch.float32,
                               kernel_size=3, stride=1, padding="same", padding_mode="reflect")

        self.bnorm = nn.BatchNorm2d(num_features=64)

        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)

        # # Initialize
        # with torch.no_grad():
        #     for layer in (self.conv5, self.conv6):
        #         nn.init.dirac_(layer.weight)
        #         dirac = torch.clone(layer.weight)
        #         nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
        #         noise = torch.clone(layer.weight)
        #         layer.weight = nn.Parameter(noise + dirac)

    def forward(self, x):
        h0 = self.relu1(self.conv1(x))
        h1 = self.relu2(self.conv2(h0))
        h2 = self.relu3(self.conv3(h1))
        h2 = self.bnorm(h2)
        h3 = self.relu4(x + self.conv4(h2))
        h4 = self.relu5(self.conv5(h3))
        y = self.conv6(h4)
        return y


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


# =========================================================================== #
#                           TRANSFORM LAYERS
# --------------------                                   -------------------- #


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


class ConvertDCTtoRGB(torch.nn.Module):
    
    def __init__(self, luma_mean=None, luma_std=None,
                 chroma_mean=None, chroma_std=None):
        super().__init__()
        self.idct = InverseDCT(luma_mean, luma_std, chroma_mean, chroma_std)
        self.to_rgb = ConvertYccToRGB()
    
    def forward(self, x, chroma):
        return self.to_rgb(self.idct(x, chroma))


class Blockify(torch.nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x, flatten=False):
        N, C, H, W = x.shape
        k = self.block_size
        pad_h = int(math.ceil(H / k) * k) - H
        pad_w = int(math.ceil(W / k) * k) - W
        num_blocks_h = (H + pad_h) // k
        num_blocks_w = (W + pad_w) // k
        canvas = torch.nn.functional.pad(x, (0, pad_h, 0, pad_w), mode="replicate")
        canvas = canvas.reshape(N, C, num_blocks_h, k, num_blocks_w, k)
        res = canvas.permute(0, 1, 2, 4, 3, 5)
        if flatten:
            return res.reshape(N, C, num_blocks_h, num_blocks_w, k ** 2)
        else:
            return res.reshape(N, C, num_blocks_h, num_blocks_w, k, k)


# class Deblockify(torch.nn.Module):

#     def __init__(self, block_size):
#         super().__init__()
#         self.block_size = block_size

#     def forward(self, x):
#         N, C, H, W, L = x.shape


# class DiscreteCosineTransform(torch.nn.Module):


#     def __init__(self, luma_mean=None, luma_std=None,
#                  chroma_mean=None, chroma_std=None):

#         super().__init__()

#         # Make Type III harmonics
#         steps = torch.arange(8, requires_grad=False) / 16
#         f = 2 * torch.arange(8, requires_grad=False) + 1
#         h1 = torch.cos(torch.outer(steps, f * torch.pi))
#         h2 = h1.clone()
#         # Make IDCT basis
#         basis = h1.T.view(8, 1, 8, 1) * h2.T.view(1, 8, 1, 8)
#         self.register_buffer("basis", basis)        # (8,8, 8,8)
#         # Make normalization matrix
#         c = torch.ones(8, dtype=torch.float32)
#         c[0] = torch.sqrt(torch.tensor(0.5, dtype=torch.float32))
#         C = 0.25 * torch.outer(c, c)
#         self.register_buffer("scale", C)

#         if luma_mean is None:
#             self.register_buffer("luma_mean", torch.zeros(64).float())
#         else:
#             self.register_buffer("luma_mean", torch.as_tensor(luma_mean).float())

#         if luma_std is None:
#             self.register_buffer("luma_std", torch.ones(64).float())
#         else:
#             self.register_buffer("luma_std", torch.as_tensor(luma_std).float())

#         if chroma_mean is None:
#             self.register_buffer("chroma_mean", torch.zeros(64).float())
#         else:
#             self.register_buffer("chroma_mean", torch.as_tensor(chroma_mean).float())

#         if chroma_std is None:
#             self.register_buffer("chroma_std", torch.ones(64).float())
#         else:
#             self.register_buffer("chroma_std", torch.as_tensor(chroma_std).float())

#     def forward(self, dct: torch.FloatTensor, chroma: bool):
#         # (B, 64, H, W)
#         B, C, H, W = dct.shape
#         assert C == 64
#         # Normalize
#         if chroma:
#             out = dct * self.chroma_std.view(1, 64, 1, 1) + self.chroma_mean.view(1, 64, 1, 1)
#         else:
#             out = dct * self.luma_std.view(1, 64, 1, 1) + self.luma_mean.view(1, 64, 1, 1)
#         # Reshape
#         out = out.view(B, 1, 1, 8, 8, H, W)
#         C = self.scale.view(1, 1, 1, 8, 8, 1, 1)
#         basis = self.basis.view(1, 8, 8, 8, 8, 1, 1)
#         # Apply transform
#         res = torch.sum(C * basis * out, dim=(3,4))         # (B, 8, 8, H, W)
#         res = res.permute(0, 3, 1, 4, 2).contiguous()       # (B, H, 8, W, 8)

#         return (res.view(B, 1, 8*H, 8*W) + 128) / 255.0


class ChromaCrop(torch.nn.Module):

    def __init__(self, height: int, width: int):
        self.height = height
        self.width  = width

    def forward(self, x: torch.FloatTensor):
        return x[:, :, :self.height, :self.width]


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


class Local2DAttentionLayer(nn.Module):

    def __init__(self, kernel_size: int, embed_dim: int, num_heads: int,
                 bias=True, add_bias_kqv=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert kernel_size % 2 == 1

        self.kernel_size = kernel_size
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.bias        = bias
        self.add_bias_kv = add_bias_kqv

        # Local neighborhood patch unfold
        p = kernel_size // 2
        self.unfold = nn.Unfold(kernel_size, padding=(p, p), stride=1)
        # self.project_q = nn.Linear(embed_dim, embed_dim, bias=add_bias_kqv)
        # self.project_k = nn.Linear(embed_dim, embed_dim, bias=add_bias_kqv)
        # self.project_v = nn.Linear(embed_dim, embed_dim, bias=add_bias_kqv)

        conv_kwargs = dict(in_channels=embed_dim, out_channels=embed_dim,
                           kernel_size=1, stride=1, padding=0, bias=add_bias_kqv)
        # Q, K, V projections
        self.project_q = nn.Conv2d(**conv_kwargs)
        self.project_k = nn.Conv2d(**conv_kwargs)
        self.project_v = nn.Conv2d(**conv_kwargs)

        # Since no nonlinearity follows the projection matrices, we'll
        # initialize them with Xavier's method
        gain = 1.0
        torch.nn.init.xavier_uniform_(self.project_q.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.project_k.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.project_v.weight, gain=gain)

        # Relative positional encodings bias table
        self.relative_positional_bias = nn.Parameter(
            torch.zeros((num_heads, 1, kernel_size ** 2), dtype=torch.float32)
        )

    def forward(self, x):
        """
        Parameters:
        ----------
            x: torch.Tensor[N, E, H, W]
        """
        N, E, H, W = x.shape
        P = self.kernel_size ** 2
        L = H * W
        Nh = self.num_heads
        dk = self.embed_dim // self.num_heads

        # Project pixels / blocks into K, Q, V
        keys   = self.project_k(x)
        query  = self.project_q(x)
        values = self.project_v(x)
        # values = x

        # Then, extract local neighborhood patches for each
        # pixel / block in `keys` and `values`
        K = self.unfold(keys).reshape(N, E, P, L).contiguous().view(N, Nh, dk, P, L)
        V = self.unfold(values).reshape(N, E, P, L).contiguous().view(N, Nh, dk, P, L)
        Q = query.view(N, E, 1, L).contiguous().view(N, Nh, dk, 1, L)

        # K should be permuted to (N, L, Nh, P, dk)
        # Q should be permuted to (N, L, Nh, 1, dk)
        # V should be permuted to (N, L, Nh, P, dk)
        K = K.permute(0,4,1,3,2)
        Q = Q.permute(0,4,1,3,2)
        V = V.permute(0,4,1,3,2)

        attn_weights = Q @ K.transpose(3, 4) / math.sqrt(self.embed_dim)
        # attn_weights.shape == (N, L, Nh, 1, P)
        attn_scores = attn_weights + self.relative_positional_bias.unsqueeze(0)
        attn = nn.functional.softmax(attn_scores, dim=-1)

        # (N, L, Nh, 1, P) @ (N, L, Nh, P, dk) -> (N, L, Nh, 1, dk)
        out = (attn @ V).squeeze(dim=3)     # (N, L, Nh, dk)
        return out.reshape(N, H, W, E).permute(0,3,1,2), attn_scores


class SpectralEncoderLayer(nn.Module):
    """Expects input of shape (N, H, W, E)"""

    def __init__(self, window_size=7, d_model=128, num_heads=4,
                 d_feedforward=512, dropout=0.1, activation=nn.GELU, bias=True,
                 layer_norm_eps=1e-05, add_bias_kqv=True):

        super().__init__()

        self.window_size    = window_size
        self.d_model        = d_model
        self.num_heads      = num_heads
        self.d_feedforward  = d_feedforward
        self.dropout        = dropout
        self.activation     = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias           = bias
        self.add_bias_kv    = add_bias_kqv

        self.local_attention = Local2DAttentionLayer(
            kernel_size=window_size,
            embed_dim=d_model,
            num_heads=num_heads,
            bias=bias,
            add_bias_kqv=add_bias_kqv,
        )
        self.layernorm1  = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layernorm2  = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.activation  = activation
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_feedforward),
            activation(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_feedforward, out_features=d_model),
        )

    def forward(self, x: torch.tensor):
        # N, H, W, E = x.shape
        y0 = self.layernorm1(x)                                 # (N, H, W, E)
        y1, _ = self.local_attention(y0.permute(0, 3, 1, 2))    # (N, E, H, W)
        y2 = y1.permute(0, 2, 3, 1)                             # (N, H, W, E)
        # return x + y2
        y3 = self.layernorm2(y2 + x)                            # (N, H, W, E)
        return y2 + self.feedforward(y3)                        # (N, E, H, W)


class SpectralEncoder(nn.Module):

    def __init__(self, in_features=64, num_layers=4, window_size=7, d_model=128,
                 d_qcoeff=64, num_heads=4, d_feedforward=1024, dropout=0.1,
                 activation=nn.GELU, bias=True, add_bias_kqv=True
        ):
        super().__init__()

        self.num_layers = num_layers
        self.embed_dim  = d_model

        # Input Embedding Layer
        self.input_embedding = nn.Linear(
            in_features=in_features, out_features=d_model, bias=False)

        self.positional_embedding = nn.Linear
        encoders = [
            SpectralEncoderLayer(window_size, d_model, d_qcoeff, num_heads,
                                    d_feedforward, dropout, activation, bias,
                                    add_bias_kqv=add_bias_kqv)
                for _ in range(num_layers)
        ]
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x, qcoeff):
        x = x.permute(0, 2, 3, 1)           # (N, H, W, C)
        emb = self.input_embedding(x)       # (N, H, W, E)
        z = emb.permute(0, 3, 1, 2)
        for encoder in self.encoders:
            z = encoder(z, qcoeff)
        out = z
        return out



class SpectralNet(nn.Module):

    def __init__(self, blocks: List[nn.Module], output_transform: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.output_transform = output_transform

    def forward(self,
                dct_tensor: torch.FloatTensor,
                qt_tensor: torch.FloatTensor,
                chroma: bool
        ):
        x = dct_tensor
        for block in self.blocks:
            if isinstance(block, SpectralEncoder):
                x = block(x, qt_tensor)
            elif isinstance(block, nn.Conv2d):
                x = block(x)
            elif isinstance(block, InverseDCT):
                x = block(x, chroma)
        out = self.output_transform(dct_tensor + x, chroma)
        return out


class BlockEncoder(nn.Module):
    """Encodes DCT tensor with shape (N, 64, H, W) to embedding vectors with
    shape (N, H, W, E)."""

    def __init__(self, in_channels: int, out_channels: int, interaction: str):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert interaction in ("FiLM", "CFM", "concat", "none")
        self.interaction = interaction
        if interaction == "FiLM":
            self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, "same")
            self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1)
        elif interaction == "CFM":
            raise NotImplementedError
        elif interaction == "concat":
            self.project = nn.Linear(in_channels + DIM_QTABLE, out_channels)
        elif interaction == "none":
            self.project = nn.Conv2d(in_channels, out_channels, 1, 1)

    def _forward_film(self, dctensor, qt):
        N = dctensor.shape[0]
        x = qt.view(N, DIM_QTABLE, 1, 1) * self.conv1(dctensor)
        return self.conv2(dctensor + x).permute(0,2,3,1)

    def _forward_concat(self, dctensor, qt):
        N, _, H, W = dctensor.shape
        x0 = torch.tile(qt.view(N, DIM_QTABLE, 1, 1), (1, 1, H, W))
        x1 = torch.cat([dctensor, x0], dim=1)
        return self.project(x1.permute(0,2,3,1))

    def forward(self, dctensor, qt):
        # dctensor.shape == (N, 64, H, W)
        # qt.shape == (N, `DIM_QTABLE`)
        if self.interaction == "FiLM":
            return self._forward_film(dctensor, qt)
        elif self.interaction == "CFM":
            raise NotImplementedError
        elif self.interaction == "concat":
            return self._forward_concat(dctensor, qt)
        elif self.interaction == "none":
            return self.project(dctensor).permute(0,2,3,1)


class BlockDecoder(nn.Module):
    """Decodes embedding tensor with shape (N, H, W, E) to DCT residual with
    shape (N, 64, H, W)."""

    def __init__(self, in_channels: int, out_channels: int, interaction: str):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert interaction in ("FiLM", "CFM", "concat", "none")
        self.interaction = interaction
        self.project = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, embedding, qt):
        embedding = embedding.permute(0,3,1,2)
        if self.interaction == "FiLM":
            N = embedding.shape[0]
            return self.project(embedding) * qt.view(N, DIM_QTABLE, 1, 1)
        elif self.interaction == "CFM":
            raise NotImplementedError
        elif self.interaction == "concat":
            raise NotImplementedError
        elif self.interaction == "none":
            return self.project(embedding)


class SpectralTransformer(nn.Module):

    def __init__(self, encoder_layers: List[SpectralEncoderLayer]):
        super().__init__()
        self.num_layers = len(encoder_layers)
        self.encoders = nn.ModuleList(encoder_layers)

    def forward(self, x):
        for enc in self.encoders:
            x = enc(x)
        return x

class SpectralModel(nn.Module):

    def __init__(self, block_encoder: BlockEncoder, transformer: SpectralTransformer,
                block_decoder: BlockDecoder):
        super().__init__()
        self.block_encoder = block_encoder
        self.transformer = transformer
        self.block_decoder = block_decoder

    def forward(self, dct, qt):
        # dct.shape == (N, C, H, W)
        # qt.shape ==  (N, `DIM_QTABLE`)
        emb = self.block_encoder(dct, qt)       # (N, H, W, E)
        y = self.transformer(emb)               # (N, H, W, E)
        residual = self.block_decoder(y, qt)    # (N, `DIM_QTABLE`, H, W)
        return dct + residual


class ConvNeXtBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, kernel_size: int,
                 norm: str = "layer"):
        super().__init__()

        self.in_channels  = in_channels
        self.mid_channels = mid_channels
        self.out_channels = in_channels
        self.kernel_size  = kernel_size

        NormLayer = nn.BatchNorm2d if norm == "layer" else LayerNorm2D
        self.dwconv = nn.Conv2d(
            in_channels=    in_channels,
            out_channels=   in_channels,
            kernel_size=    kernel_size,
            padding=        "same",
            groups=         in_channels
        )
        self.norm = NormLayer(in_channels)
        self.pwconv1 = nn.Linear(in_channels, mid_channels)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        # x.shape == (B,C,H,W)
        z = x
        y = self.dwconv(z)                      # (B, C, H, W)
        y = self.norm(y).permute(0,2,3,1)
        y = self.pwconv1(y)
        y = self.activation(y)
        y = self.pwconv2(y).permute(0,3,1,2)    # (B, C, H, W)
        return z + y


class ChromaNet(nn.Module):

    def __init__(
            self,
            stages: List[ConvNeXtBlock],
            output_transform: nn.Module,
            in_channels: int = 3,
            out_channels: int = 3,
            kernel_size: int = 3,
            skip: bool = False,
            norm: str = "layer"
    ):
        super().__init__()

        self.in_channels =  in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.output_transform = output_transform
        self.skip = skip

        NormLayer = nn.BatchNorm2d if norm == "batch" else LayerNorm2D

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stages[0].in_channels, kernel_size, padding="same"),
            NormLayer(stages[0].out_channels)
        )
        body = []
        for i in range(len(stages)):
            in_c = stages[i].out_channels
            out_c = out_channels if i == len(stages) - 1 else stages[i+1].in_channels
            body.append(stages[i])
            body.append(NormLayer(in_c))
            body.append(nn.Conv2d(in_c, out_c, 1, bias=False))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        if self.skip:
            return self.output_transform(x)
        else:
            y = self.stem(x)
            y = self.body(y)
            return self.output_transform(x + y)


class LayerNorm2D(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x.shape == (N, C, H, W)
        y = self.norm(x.permute(0,2,3,1))
        return y.permute(0, 3, 1, 2)