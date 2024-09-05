import torch
import torch.nn as nn


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
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=9,
                               stride=1, padding="same", bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7,
                               stride=1, padding="same", bias=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1,
                               stride=1, padding="same", bias=False)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5,
                               stride=1, padding="same", bias=False)

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

    def forward(self, x):
        h0 = self.relu1(self.conv1(x))
        h1 = self.relu2(self.conv2(h0))
        h2 = self.relu3(self.conv3(h1))
        y  = self.conv4(h2)
        return y

    def flops(self, x):
        pass
