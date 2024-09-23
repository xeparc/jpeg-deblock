import torch
import torch.nn as nn
import torch.nn.functional as F

# class DenseLayer(nn.Module):
#     def __init__(self, in_channels, growth_rate):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=True)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.relu(out)
#         return torch.cat([x, out], 1)


# class QFAttention(nn.Module):
#     def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
#         super(QFAttention, self).__init__()

#         assert in_channels == out_channels, 'Only support in_channels==out_channels.'
#         if mode[0] in ['R', 'L']:
#             mode = mode[0].lower() + mode[1:]

#         self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

#     def forward(self, x, gamma, beta):
#         gamma = gamma.unsqueeze(-1).unsqueeze(-1)
#         beta = beta.unsqueeze(-1).unsqueeze(-1)
#         res = (gamma)*self.res(x) + beta
#         return x + res

# class ResidualDenseBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.growth_rate = growth_rate
#         self.layers = self.make_dense_layers(in_channels, growth_rate, num_layers)
#         self.conv1x1 = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1, padding=0, bias=True)

#     def make_dense_layers(self, in_channels, growth_rate, num_layers):
#         layers = []
#         for i in range(num_layers):
#             layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.layers(x)
#         out = self.conv1x1(out)
#         return out + x

# Example usage:
# Creating an RDB with 5 dense layers, each with a growth rate of 32
# rdb = ResidualDenseBlock(in_channels=64, growth_rate=32, num_layers=5)


# def make_layer(block, n_layers):
#     layers = []
#     for _ in range(n_layers):
#         layers.append(block())
#     return nn.Sequential(*layers)


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


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialization
        init_kwargs = dict(nonlinearity="leaky_relu", a=0.2)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, **init_kwargs)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, **init_kwargs)
        torch.nn.init.kaiming_uniform_(self.conv3.weight, **init_kwargs)
        torch.nn.init.kaiming_uniform_(self.conv4.weight, **init_kwargs)
        torch.nn.init.kaiming_uniform_(self.conv5.weight, **init_kwargs)
        # ! IMPORTANT ! Scale weights by factor of 0.1
        self.conv1.weight.data *= 0.1
        self.conv2.weight.data *= 0.1
        self.conv3.weight.data *= 0.1
        self.conv4.weight.data *= 0.1
        self.conv5.weight.data *= 0.1

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDB4(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, in_channels, growth=32):
        super(RRDB, self).__init__()
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
            nn.Linear(condition_dim, features_dim), nn.Sigmoid)
        self.f_beta  = nn.Sequential(
            nn.Linear(condition_dim, features_dim), nn.Tanh)

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
        gamma = self.f_gamma(condition).view(1, -1, 1, 1)
        beta  = self.f_beta(condition).view(1, -1, 1, 1)
        return gamma * features + beta

