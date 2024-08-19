import tempfile

import numpy as np
import torch
from jpeglib import read_spatial, read_dct
from turbojpeg import (
    TurboJPEG,
    TJCS_RGB, TJFLAG_ACCURATEDCT,
    TJSAMP_444, TJSAMP_441, TJSAMP_420, TJSAMP_411, TJSAMP_422, TJSAMP_440
)


DEFAULT_QTABLE_Y = torch.tensor(
   [[ 16,  11,  10,  16,  24,  40,  51,  61],
    [ 12,  12,  14,  19,  26,  58,  60,  55],
    [ 14,  13,  16,  24,  40,  57,  69,  56],
    [ 14,  17,  22,  29,  51,  87,  80,  62],
    [ 18,  22,  37,  56,  68, 109, 103,  77],
    [ 24,  35,  55,  64,  81, 104, 113,  92],
    [ 49,  64,  78,  87, 103, 121, 120, 101],
    [ 72,  92,  95,  98, 112, 100, 103,  99]], dtype=torch.float32
)

DEFAULT_QTABLE_C = torch.tensor(
   [[ 17,  18,  24,  47,  99,  99,  99,  99],
    [ 18,  21,  26,  66,  99,  99,  99,  99],
    [ 24,  26,  56,  99,  99,  99,  99,  99],
    [ 47,  66,  99,  99,  99,  99,  99,  99],
    [ 99,  99,  99,  99,  99,  99,  99,  99],
    [ 99,  99,  99,  99,  99,  99,  99,  99],
    [ 99,  99,  99,  99,  99,  99,  99,  99],
    [ 99,  99,  99,  99,  99,  99,  99,  99]], dtype=torch.float32
)

class InverseDCT(torch.nn.Module):

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


class YCbCr2RGB(torch.nn.Module):

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

class RGB2YCbCr(torch.nn.Module):
    
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


class UpsampleChrominance(torch.nn.Module):

    def __init__(self, subsample):
        super().__init__()
        subsample_factors = {
            "444": (1,1),
            "440": (2,1),
            "422": (1,2),
            "411": (1,4),
            "420": (2,2),
        }
        self.repeats = subsample_factors[subsample]

    def forward(self, x):
        # (B, C, H, W)
        y = torch.repeat_interleave(x, self.repeats[0], dim=2)
        y = torch.repeat_interleave(y, self.repeats[1], dim=3)
        return y



class JPEGImageData:

    subsample_factors = {
        "444": (1,1),
        "440": (2,1),
        "422": (1,2),
        "411": (1,4),
        "420": (2,2),
    }

    def __init__(self, rgb, yuv=None, dct=None, qt=None, quality=100,
                 subsample="422"):
        assert rgb.ndim == 3
        self.height     = rgb.shape[0]
        self.width      = rgb.shape[1]
        self.channels   = rgb.shape[2]
        assert self.channels == 3
        self.quality    = quality
        self.subsample  = subsample

        if yuv is None or dct is None:
            dat = JPEGImageData._get_jpeg_data(rgb, quality=100, subsample=subsample)
        else:
            dat = {"rgb": rgb, "yuv": yuv, "dct": dct}

        self.rgb        = torch.from_numpy(dat["rgb"]).permute(2,0,1)
        self.Y          = torch.from_numpy(dat["yuv"][0])
        self.Cb         = torch.from_numpy(dat["yuv"][1])
        self.Cr         = torch.from_numpy(dat["yuv"][2])
        self.dctY       = torch.from_numpy(dat["dct"][0])
        self.dctCb      = torch.from_numpy(dat["dct"][1])
        self.dctCr      = torch.from_numpy(dat["dct"][2])

    @staticmethod
    def read(image_path):
        pass

    def get_upsampled_yuv(self):
        # Upsample Cb, Cr planes
        h, w = JPEGImageData.subsample_factors[self.subsample]
        u = np.repeat(np.repeat(self.Cb, h, axis=0), w, axis=1)
        v = np.repeat(np.repeat(self.Cr, h, axis=0), w, axis=1)
        u = u[:self.height, :self.width]
        v = v[:self.height, :self.width]
        return torch.stack([self.Y, u, v], dim=0)

    def _get_jpeg_data(rgb, quality=100, subsample="422"):
        # Assert that image is color and channel dimension is last
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3

        f = tempfile.NamedTemporaryFile(mode="w+b", delete=True)
        turbo = TurboJPEG()
        subsampledict = {
            "444": TJSAMP_444,
            "440": TJSAMP_440,
            "422": TJSAMP_422,
            "411": TJSAMP_411,
            "420": TJSAMP_420,
        }

        smode = subsampledict[subsample]

        buff = turbo.encode(rgb, quality, pixel_format=TJCS_RGB,
                            jpeg_subsample=smode, flags=TJFLAG_ACCURATEDCT)
        f.write(buff)
        f.flush()

        dctobj = read_dct(f.name)
        f.close()
        yuv = turbo.decode_to_yuv_planes(buff, flags=TJFLAG_ACCURATEDCT)
        qtb = [dctobj.qt[0], dctobj.qt[1]]
        if len(dctobj.qt < 3):
            qtb.append(dctobj.qt[1])
        else:
            qtb.append(dctobj.qt[2])
        dct = [dctobj.Y  * qtb[0], dctobj.Cb * qtb[1], dctobj.Cr * qtb[2]]

        return {
            "rgb": rgb.transpose(2, 0, 1),      # RGB data
            "yuv": yuv,                         # Y, Cb, Cr planes (downsampled)
            "dct": dct                          # DCT of Y, Cb, Cr planes
        }

    @torch.no_grad()
    def quantize(self, quality: int):
        """
        Quantize DCT and return copy of `self` with updated spatial
        and frequency attributes.
        """
        # Obtain quantization tables for selected JPEG quality
        qtab_Y = get_Y_qtable(quality).float()
        qtab_C = get_C_qtable(quality).float()

        # Do the quantization. Range of values after that = [0, 2**16]
        q_dctY  = torch.round(self.dctY  / qtab_Y) * qtab_Y
        q_dctCb = torch.round(self.dctCb / qtab_C) * qtab_C
        q_dctCr = torch.round(self.dctCr / qtab_C) * qtab_C

        # Restore Y, Cb, Cr channels from quantized DCT data
        # Range of values after that = [0, 2**16-1]
        bY  = q_dctY.view(1, q_dctY.shape[0],  q_dctY.shape[1], 64).permute(0,3,1,2)
        bCb = q_dctCb.view(1, q_dctCb.shape[0], q_dctCb.shape[1], 64).permute(0,3,1,2)
        bCr = q_dctCr.view(1, q_dctCr.shape[0], q_dctCr.shape[1], 64).permute(0,3,1,2)

        idct = InverseDCT()
        q_Y  = idct(bY)[0,0]
        q_Cb = idct(bCb)[0,0]
        q_Cr = idct(bCr)[0,0]

        # Upsample Cb, Cr planes in order to convert to RGB
        upsample_transform = UpsampleChrominance(self.subsample)
        chrominance = torch.stack([q_Cb, q_Cr], dim=0)
        chrominance = upsample_transform(chrominance.unsqueeze(0))
        q_Cb_upsampled = chrominance[0,0]
        q_Cr_upsampled = chrominance[0,1]

        # Crop
        Y = q_Y[:self.height, :self.width]
        U = q_Cb_upsampled[:self.height, :self.width]
        V = q_Cr_upsampled[:self.height, :self.width]
        upsampled_yuv = torch.stack([Y, U, V], dim=0).contiguous()

        # Restore RGB channels from YUV
        yuv2rgb = YCbCr2RGB()
        rgb = torch.round(255 * yuv2rgb(upsampled_yuv.unsqueeze(0)))[0]

        rgb = rgb.permute(1,2,0).contiguous().clip(min=0, max=255).to(dtype=torch.uint8).numpy()
        yuv = [
            torch.round((255 * q_Y).to(dtype=torch.int16)).numpy(),
            torch.round((255 * q_Cb).to(dtype=torch.int16)).numpy(),
            torch.round((255 * q_Cr).to(dtype=torch.int16)).numpy()
        ]
        dct = [
            q_dctY.to(dtype=torch.int16).numpy(),
            q_dctCb.to(dtype=torch.int16).numpy(),
            q_dctCr.to(dtype=torch.int16).numpy()
        ]

        return type(self)(rgb=rgb, yuv=yuv, dct=dct, subsample=self.subsample)



def jpeg_quality_scaling(quality):
    if (quality <= 0):
        quality = 1
    if (quality > 100):
        quality = 100
    if (quality < 50):
        quality = 5000 / quality
    else:
        quality = 200 - quality * 2
    return quality


def get_Y_qtable(quality):
    scale = jpeg_quality_scaling(quality)
    temp = (DEFAULT_QTABLE_Y * scale + 50) / 100
    temp = torch.clip(temp, min=1, max=255)
    return temp.to(dtype=torch.uint8)


def get_C_qtable(quality):
    scale = jpeg_quality_scaling(quality)
    temp = (DEFAULT_QTABLE_C * scale + 50) / 100
    temp = torch.clip(temp, min=1, max=255)
    return temp.to(dtype=torch.uint8)





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

