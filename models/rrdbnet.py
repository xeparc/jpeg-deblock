import torch
import torch.nn as nn

from .blocks import RRDB


class LumiNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, num_blocks=5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.body = nn.Sequential(
            *[RRDB(64) for _ in range(num_blocks)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.body(y)
        y = self.decoder(y)
        return y


class ChromaNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, num_blocks=3):
        super().__init__()

        self.first = nn.Conv3d(in_channels, 64, kernel_size=(1,3,3),
                               stride=1, padding=(0,1,1))

        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.body = nn.Sequential(
            *[RRDB(64) for _ in range(num_blocks)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        N, C, H, W = x.shape
        # Add depth dimension
        x = x.view(N, C, 1, H, W)
        y = self.first(x)
        # Remove depth dimension
        y = y.squeeze(dim=2)
        y = self.encoder(y)
        y = self.body(y)
        y = self.decoder(y)
        return y


class RRDBNet(nn.Module):

    def __init__(self, luma_blocks=5, chroma_blocks=3):
        super().__init__()

        self.lumi = LumiNet(in_channels=1, out_channels=1, num_blocks=luma_blocks)
        self.chroma = ChromaNet(in_channels=3, out_channels=2, num_blocks=chroma_blocks)

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == 3
        # Extract luma and chroma components
        Y, CbCr = x[:, :1, :, :], x[:, 1:, :, :]
        # Restore luma component
        Y_r = self.lumi(Y)
        # Restore chroma components
        y = torch.cat([Y_r, CbCr], dim=1)
        CbCr_r = self.chroma(y)
        # Concatenate restored components
        result = torch.cat([Y_r, CbCr_r], dim=1)
        return result



# class RRDBNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(RRDBNet, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         #### upsampling
#         self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         fea = self.conv_first(x)
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         fea = fea + trunk

#         fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         out = self.conv_last(self.lrelu(self.HRconv(fea)))

#         return out
