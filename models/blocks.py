import torch
import torch.nn as nn


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
        torch.nn.init.kaiming_uniform_(self.dsconv[1].weight, nonlinearity="leaky_relu", a=0.2)
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
