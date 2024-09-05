import torch
import torchvision


class PixelwiseFeatures(torch.nn.Module):
    """
    Implementation of

    "Pixelwise JPEG Compression Detection and Quality Factor Estimation
     Based on Convolutional Neural Network"

    paper in PyTorch. The last layer is missing, because this module is
    intended to be used as a feature extractor for perceptual loss, hence
    it's not needed.

    https://github.com/kuchida/PixelwiseJPEGCompressionDetection/
    """

    TRAINED_WEIGHTS = "../data/params/pixelwise/statedict.pth"

    def __init__(self, pretrained=False):
        super().__init__()
        kwargs = dict(kernel_size=3, padding="same")
        self.conv2d_1 = torch.nn.Conv2d( 3, 64, dilation=1, **kwargs)
        self.conv2d_2 = torch.nn.Conv2d(64, 64, dilation=2, **kwargs)
        self.conv2d_3 = torch.nn.Conv2d(64, 64, dilation=3, **kwargs)
        self.conv2d_4 = torch.nn.Conv2d(64, 64, dilation=4, **kwargs)
        self.conv2d_5 = torch.nn.Conv2d(64, 64, dilation=3, **kwargs)
        self.conv2d_6 = torch.nn.Conv2d(64, 64, dilation=2, **kwargs)
        self.conv_layers = [self.conv2d_1, self.conv2d_2, self.conv2d_3,
                            self.conv2d_4, self.conv2d_5, self.conv2d_6]
        self.num_layers = 6

        if pretrained:
            state_dict = torch.load(PixelwiseFeatures.TRAINED_WEIGHTS,
                                    weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            x = torch.nn.functional.relu(x)
        return x


class MobileNetV2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        mobi = torchvision.models.mobilenet_v2()
        self.features = mobi.features
        self.head = torch.nn.LazyLinear(1)

    def forward(self, x):
        features = self.features(x)
        vec = features.mean(dim=(2,3))
        return self.head(vec)