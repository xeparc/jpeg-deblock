import tempfile

import torch
import torch.nn as nn
from turbojpeg import (
    TurboJPEG,
    TJCS_RGB, TJFLAG_ACCURATEDCT,
    TJSAMP_444, TJSAMP_441, TJSAMP_420, TJSAMP_411, TJSAMP_422, TJSAMP_440
)
from jpeglib import read_dct, read_spatial


# Define basis for IDCT transform

class InverseDCT(nn.Module):

    def __init__(self, mean=None, std=None):
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

        if mean is None:
            self.register_buffer("mean", torch.zeros(64, dtype=torch.float32))
        else:
            self.register_buffer("mean", mean.float())

        if std is None:
            self.register_buffer("std", torch.ones(64, dtype=torch.float32))
        else:
            self.register_buffer("std", std.float())

    def forward(self, dct):
        # (B, 64, H, W)
        B, C, H, W = dct.shape
        assert C == 64
        # Normalize
        dct = dct * self.std.view(1, 64, 1, 1) + self.mean.view(1, 64, 1, 1)
        # Reshape
        dct = dct.view(B, 1, 1, 8, 8, H, W)
        C = self.scale.view(1, 1, 1, 8, 8, 1, 1)
        basis = self.basis.view(1, 8, 8, 8, 8, 1, 1)
        res = torch.sum(C * basis * dct, dim=(3,4))         # (B, 8, 8, H, W)
        res = res.permute(0, 3, 1, 4, 2).contiguous()       # (B, H, 8, W, 8)

        return (res.view(B, 1, 8*H, 8*W) + 128.0) / 255.0



class JPEGData:

    def __init__(self, rgb, Y, Cb, Cr, dctY, dctCb, dctCr, qtY, qtC, subsample):
        assert rgb.ndim == 3
        assert qtY.ndim == 2
        assert qtC.ndim == 2
        self.rgb        = torch.from_numpy(rgb)
        self.Y          = torch.from_numpy(Y)
        self.Cb         = torch.from_numpy(Cb)
        self.Cr         = torch.from_numpy(Cr)
        self.dctY       = torch.from_numpy(dctY)
        self.dctCb      = torch.from_numpy(dctCb)
        self.dctCr      = torch.from_numpy(dctCr)
        self.qtY        = torch.from_numpy(qtY)
        self.qtC        = torch.from_numpy(qtC)
        self.subsample  = subsample



def get_jpeg_data(img, quality=90, subsample="422"):
    # Assert that image is color and channel dimension is last
    assert img.ndim == 3
    assert img.shape[2] == 3

    f = tempfile.NamedTemporaryFile(mode="w+b", delete=True)
    turbo = TurboJPEG()
    subsampledict = {
        "422": TJSAMP_444,
        "440": TJSAMP_440,
        "441": TJSAMP_441,
        "422": TJSAMP_422,
        "411": TJSAMP_411,
        "420": TJSAMP_420,
    }
    smode = subsampledict[subsample]
    buff = turbo.encode(img, quality, pixel_format=TJCS_RGB,
                        jpeg_subsample=smode, flags=TJFLAG_ACCURATEDCT)
    f.write(buff)
    f.flush()

    dctobj = read_dct(f.name)
    f.close()
    yuv = turbo.decode_to_yuv_planes(buff)

    return JPEGData(
        img,                                # RGB data
        yuv[0], yuv[1], yuv[2],             # Y, Cb, Cr planes
        dctobj.Y * dctobj.qt[0],            # Dequantized DCT Y
        dctobj.Cb * dctobj.qt[1],           # Dequantized DCT Cb
        dctobj.Cr * dctobj.qt[1],           # Dequantized DCT Cr
        dctobj.qt[0], dctobj.qt[1],         # Y, Cr/Cb quantization tables
        subsample=subsample                 # subsampling mode
    )


# class JPEGEncoder:

#     def __init__(self, mode="4:2:2"):
#         assert mode in ("4:2:0", "4:2:2")
#         self.mode = str(mode)
    
#     def pad_YCbCr_planes(self, Y, Cb, Cr):
#         H, W = Y.shape
#         Hp = int(np.ceil(H / 16) * 16)
#         Wp = int(np.ceil(W / 16) * 16)
#         pad_left = (Wp - H) // 2
#         pad_right = Wp - pad_left
#         pad_top = (Hp - H) // 2
#         pad_bottom = Hp - pad_top
#         Cr = np.pad(Cr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
#         Cb = np.pad(Cb, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")

