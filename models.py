from typing import List

import numpy as np
import torch
import torch.nn as nn

from jpegutils import SUBSAMPLE_FACTORS


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


class Local2DAttentionLayer(nn.Module):

    def __init__(self, kernel_size=7, embed_dim=128, num_heads=4, bias=True,
                 add_bias_kqv=True):
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
        # self.project_q = nn.LazyLinear(out_features=embed_dim, bias=True)
        # self.project_k = nn.LazyLinear(out_features=embed_dim, bias=True)
        # self.project_v = nn.LazyLinear(out_features=embed_dim, bias=True)

        conv_kwargs = dict(in_channels=embed_dim, out_channels=embed_dim,
                           kernel_size=1, stride=1, padding=0, bias=add_bias_kqv)
        # Q, K, V projections
        self.project_q = nn.Conv2d(**conv_kwargs)
        self.project_k = nn.Conv2d(**conv_kwargs)
        self.project_v = nn.Conv2d(**conv_kwargs)

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
        Hd = self.embed_dim // self.num_heads

        # Project pixels / blocks into K, Q, V
        keys   = self.project_k(x)
        query  = self.project_q(x)
        values = self.project_v(x)

        # Then, extract local neighborhood patches for each
        # pixel / block in `keys` and `values`
        K = self.unfold(keys).reshape(N, E, P, L).contiguous().view(N, Nh, Hd, P, L)
        V = self.unfold(values).reshape(N, E, P, L).contiguous().view(N, Nh, Hd, P, L)
        Q = query.view(N, E, 1, L).contiguous().view(N, Nh, Hd, 1, L)

        # K should be permuted to (N, L, Nh, P, Hd)
        # Q should be permuted to (N, L, Nh, 1, Hd)
        # V should be permuted to (N, L, Nh, P, Hd)
        K = K.permute(0,4,1,3,2)
        Q = Q.permute(0,4,1,3,2)
        V = V.permute(0,4,1,3,2)

        attn_weights = Q @ K.transpose(3, 4)
        # attn_weights.shape == (N, L, Nh, 1, P)
        attn_scores = attn_weights + self.relative_positional_bias.unsqueeze(0)
        attn = nn.functional.softmax(attn_scores, dim=-1)

        # (N, L, Nh, 1, P) @ (N, L, Nh, P, Hd) -> (N, L, Nh, 1, Hd)
        out = (attn @ V).squeeze(dim=3)     # (N, L, Nh, Hd)
        return out.reshape(N, H, W, E).permute(0,3,1,2)


class SpectralEncoderLayer(nn.Module):

    def __init__(self, kernel_size=7, d_model=128, d_qcoeff=64, num_heads=4,
                 d_feedforward=512, dropout=0.1, activation=nn.GELU, bias=True,
                 layer_norm_eps=1e-05, add_bias_kqv=True):

        super().__init__()

        self.kernel_size    = kernel_size
        self.d_model        = d_model
        self.d_qcoeff       = d_qcoeff
        self.num_heads      = num_heads
        self.d_feedforward  = d_feedforward
        self.dropout        = dropout
        self.activation     = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias           = bias
        self.add_bias_kv    = add_bias_kqv

        self.local_attention = Local2DAttentionLayer(
            kernel_size=kernel_size,
            embed_dim=d_model,
            num_heads=num_heads,
            bias=bias,
            add_bias_kqv=add_bias_kqv,
        )
        self.bilinear = nn.Bilinear(
            in1_features=d_model,
            in2_features=d_qcoeff,
            out_features=d_model
        )
        self.layernorm1  = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)
        self.layernorm2  = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)
        self.activation  = activation
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_feedforward),
            activation(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_feedforward, out_features=d_model),
        )

    def forward(self, x, qcoeff):
        # qcoeff is batched

        E = self.d_model
        N, C, H, W = x.shape
        qcoeff = torch.tile(qcoeff.view(N, 1, 1, self.d_qcoeff), (1, H, W, 1))

        xnorm = self.layernorm1(x.permute(0, 2, 3, 1))          # (N, H, W, E)
        z0 = self.local_attention(xnorm.permute(0, 3, 1, 2))    # (N, E, H, W)
        z0 = z0.permute(0, 2, 3, 1)                             # (N, H, W, E)
        z1 = self.bilinear(z0, qcoeff)                          # (N, H, W, E)
        z2 = z1 + x.permute(0, 2, 3, 1)
        z3 = self.layernorm2(z2)                                # (N, H, W, E)
        z4 = self.feedforward(z3) + z2                          # (N, H, W, E)
        return z4.permute(0, 3, 1, 2)


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
        return yuv.clip(min=0.0, max=1.0)


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
            self.register_buffer("chroma_std", torch.zeros(64).float())
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
        res = torch.sum(C * basis * out, dim=(3,4))         # (B, 8, 8, H, W)
        res = res.permute(0, 3, 1, 4, 2).contiguous()       # (B, H, 8, W, 8)

        return (res.view(B, 1, 8*H, 8*W) + 128.0) / 255.0


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


class ConvNeXtBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, kernel_size: int):
        super().__init__()

        self.in_channels  = in_channels
        self.mid_channels = mid_channels
        self.out_channels = in_channels
        self.kernel_size  = kernel_size

        self.dwconv = nn.Conv2d(
            in_channels=    in_channels,
            out_channels=   in_channels,
            kernel_size=    kernel_size,
            padding=        "same",
            groups=         in_channels
        )
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, mid_channels)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        # x.shape == (B,C,H,W)
        z = x
        y = self.dwconv(z).permute(0,2,3,1)     # (B, H, W, C)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = self.activation(y)
        y = self.pwconv2(y).permute(0,3,1,2)    # (B, C, H, W)
        return x + y


class ChromaNet(nn.Module):

    def __init__(
            self,
            stages: List[ConvNeXtBlock],
            output_transform: nn.Module,
            in_channels: int = 3,
            out_channels: int = 3,
            kernel_size: int = 3,
            skip = False
    ):
        super().__init__()

        self.in_channels =  in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.output_transform = output_transform
        self.skip = skip

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stages[0].in_channels, kernel_size, padding="same"),
            LayerNorm2D(stages[0].in_channels)
        )
        self.body = nn.Sequential(*stages)
        linear = []
        for i in range(len(stages)):
            in_c = stages[i].out_channels
            out_c = out_channels if i == len(stages) - 1 else stages[i+1].in_channels
            layer = nn.Sequential(
                LayerNorm2D(in_c),
                nn.Conv2d(in_c, out_c, 1, bias=False))
            linear.append(layer)
        self.linear = nn.ModuleList(linear)


    def forward(self, x):
        if self.skip:
            return self.output_transform(x)
        else:
            y = self.stem(x)
            for i in range(len(self.body)):
                y = self.body[i](y)
                y = self.linear[i](y)
            return self.output_transform(x + y)


class LayerNorm2D(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x.shape == (N, C, H, W)
        y = self.norm(x.permute(0,2,3,1))
        return y.permute(0, 3, 1, 2)