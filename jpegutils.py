from typing import List
from functools import lru_cache

import numpy as np
from scipy.fft import dctn, idctn
from turbojpeg import (
    TurboJPEG,
    TJCS_RGB, TJFLAG_ACCURATEDCT,
)


DEFAULT_QTABLE_Y = np.array(
   [[ 16,  11,  10,  16,  24,  40,  51,  61],
    [ 12,  12,  14,  19,  26,  58,  60,  55],
    [ 14,  13,  16,  24,  40,  57,  69,  56],
    [ 14,  17,  22,  29,  51,  87,  80,  62],
    [ 18,  22,  37,  56,  68, 109, 103,  77],
    [ 24,  35,  55,  64,  81, 104, 113,  92],
    [ 49,  64,  78,  87, 103, 121, 120, 101],
    [ 72,  92,  95,  98, 112, 100, 103,  99]], dtype=np.float32
)

DEFAULT_QTABLE_C = np.array(
   [[ 17,  18,  24,  47,  99,  99,  99,  99],
    [ 18,  21,  26,  66,  99,  99,  99,  99],
    [ 24,  26,  56,  99,  99,  99,  99,  99],
    [ 47,  66,  99,  99,  99,  99,  99,  99],
    [ 99,  99,  99,  99,  99,  99,  99,  99],
    [ 99,  99,  99,  99,  99,  99,  99,  99],
    [ 99,  99,  99,  99,  99,  99,  99,  99],
    [ 99,  99,  99,  99,  99,  99,  99,  99]], dtype=np.float32
)

YCC2RGB_MATRIX = np.array(
    [[1.0,  0.0,       1.40200],
     [1.0,  -0.34414, -0.71414],
     [1.0,  1.772,         0.0]], dtype=np.float32
)

RGB2YCC_MATRIX = np.array(
    [[ 0.29900,  0.58700,  0.11400],
     [-0.16874, -0.33126,  0.50000],
     [ 0.50000, -0.41869, -0.08131]], dtype=np.float32
)

SUBSAMPLE_FACTORS = {
    "444": (1,1),
    "440": (2,1),
    "422": (1,2),
    "411": (1,4),
    "420": (2,2),
}


class JPEGTransforms:

    def __init__(self, image):
        assert image.ndim == 3
        self.height     = image.shape[0]
        self.width      = image.shape[1]
        self.channels   = image.shape[2]
        assert self.channels == 3
        self.rgb = image

    @staticmethod
    def read(image_path):
        turbo = TurboJPEG()
        with open(image_path, mode="rb") as f:
            b = f.read()
            rgb = turbo.decode(b, pixel_format=TJCS_RGB, flags=TJFLAG_ACCURATEDCT)
            return rgb

    def get_rgb(self) -> np.ndarray:
        return self.rgb

    @lru_cache
    def get_ycc_planes(self, subsample: str) -> List[np.ndarray]:
        planes = rgb2ycc(self.rgb)
        return subsample_chrominance(planes, subsample=subsample)

    @lru_cache
    def get_upsampled_ycc_planes(self, subsample: str) -> List[np.ndarray]:
        planes = self.get_ycc_planes(subsample)
        upsampled = upsample_chrominance(planes, subsample)
        # Crop
        res = [plane[:self.height, :self.width] for plane in upsampled]
        return np.stack(res, axis=0)

    @lru_cache
    def get_dct_planes(self, subsample: str) -> List[np.ndarray]:
        ycc_planes = self.get_ycc_planes(subsample)
        ycc_plane_blocks = to_data_units(ycc_planes)
        result = []
        for plane in ycc_plane_blocks:
            # ! INPORTANT ! Change range [0..255] -> [-128..127]
            plane = plane.astype(np.float32) - 128.0
            plane_dct = dctn(plane, type=2, axes=(2,3), norm="ortho")
            result.append(plane_dct)
        return result

    @lru_cache
    def encode(self, quality: int, subsample) -> List[np.ndarray]:
        dct_planes = self.get_dct_planes(subsample=subsample)
        qtab_y = self.get_y_qtable(quality)
        qtab_c = self.get_c_qtable(quality)

        encoded = []
        for plane, qt in zip(dct_planes, (qtab_y, qtab_c, qtab_c)):
            x = np.round(plane / qt) * qt
            encoded.append(x.astype(np.int16))
        return encoded

    def decode_ycc(self, dct_blocks: List[np.ndarray]) -> List[np.ndarray]:
        ycc = []
        for plane_blocks in dct_blocks:
            x = plane_blocks.astype(np.float32)
            # ! INPORTANT ! Chnage range [-128..127] -> [0..255]
            plane = 128.0 + idctn(x, type=2, axes=(2,3), norm="ortho")
            plane = np.ascontiguousarray(np.transpose(plane, (0,2,1,3)))
            plane = plane.reshape(8 * plane.shape[0], 8 * plane.shape[2])
            plane = np.clip(np.round(plane), a_min=0, a_max=255)
            ycc.append(plane.astype(np.uint8))
        return ycc

    def decode_rgb(self, dct_blocks: np.ndarray, subsample) -> List[np.ndarray]:
        ycc = self.decode_ycc(dct_blocks)
        ycc_upsampled = upsample_chrominance(ycc, subsample)
        # Crop
        channels = [plane[:self.height, :self.width] for plane in ycc_upsampled]
        channels = np.transpose(np.stack(channels, axis=0), (1,2,0))
        rgb = ycc2rgb(channels)
        assert rgb.shape[0] == self.height
        assert rgb.shape[1] == self.width
        assert rgb.shape[2] == self.channels
        return np.ascontiguousarray(rgb).astype(np.uint8)

    @lru_cache
    def get_y_qtable(self, quality: int) -> np.ndarray:
        return get_qtable(quality, chrominance=False)

    @lru_cache
    def get_c_qtable(self, quality: int) -> np.ndarray:
        return get_qtable(quality, chrominance=True)


def rgb2ycc(rgb: np.ndarray) -> np.ndarray:
    assert rgb.ndim == 3
    assert rgb.shape[2] == 3
    indtype = rgb.dtype
    assert indtype == np.float32 or indtype == np.uint8

    y = np.einsum("ij,hwj->hwi", RGB2YCC_MATRIX, rgb.astype(np.float32))
    if indtype == np.float32:
        ycc = y + np.array([0.0, 128/255, 128/255], dtype=np.float32)
        ycc = np.clip(ycc, a_min=0.0, a_max=1.0)
    elif indtype == np.uint8:
        ycc = y + np.array([0, 128, 128], dtype=np.float32)
        print(ycc.max(), ycc.min())
        ycc = np.clip(np.round(ycc), a_min=0, a_max=255)
    return ycc.astype(indtype)


def ycc2rgb(ycc: np.ndarray) -> np.ndarray:
    assert ycc.ndim == 3
    assert ycc.shape[2] == 3
    indtype = ycc.dtype
    assert indtype == np.float32 or indtype == np.uint8

    offset = np.array([0, -128, -128], dtype=np.float32)
    if indtype == np.float32:
        offset /= 255.0

    ycc = ycc + offset
    rgb = np.einsum("ij,hwj->hwi", YCC2RGB_MATRIX, ycc.astype(np.float32))
    if indtype == np.float32:
        rgb = np.clip(rgb, a_min=0.0, a_max=1.0)
    elif indtype == np.uint8:
        rgb = np.clip(np.round(rgb), a_min=0, a_max=255)
    return rgb.astype(indtype)


def subsample_chrominance(ycc: np.ndarray, subsample: str) -> List[np.ndarray]:
    Y, Cb, Cr = np.moveaxis(ycc, 2, 0)
    assert Cb.shape == Cr.shape
    h, w = Cb.shape
    h_stride, w_stride = SUBSAMPLE_FACTORS[subsample]
    # Pad - the image height and width may not be multiple of subsample factors
    pad_h = int(np.ceil(h / h_stride) * h_stride) - h
    pad_w = int(np.ceil(w / w_stride) * w_stride) - w
    cb = np.pad(Cb, [(0, pad_h), (0, pad_w)], mode="edge")
    cr = np.pad(Cr, [(0, pad_h), (0, pad_w)], mode="edge")
    # Subsample
    cb = cb[::h_stride, ::w_stride]
    cr = cr[::h_stride, ::w_stride]
    return [Y, cb, cr]


def upsample_chrominance(ycc: List[np.ndarray], subsample: str) -> List[np.ndarray]:
    Y, Cb, Cr = ycc
    h_repeat, w_repeat = SUBSAMPLE_FACTORS[subsample]
    cb = np.repeat(np.repeat(Cb, h_repeat, axis=0), w_repeat, axis=1)
    cr = np.repeat(np.repeat(Cr, h_repeat, axis=0), w_repeat, axis=1)
    return [Y, cb, cr]


def to_data_units(ycc: List[np.ndarray]) -> List[np.ndarray]:
    return [split_plane_to_blocks(plane) for plane in ycc]


def split_plane_to_blocks(plane: np.ndarray) -> np.ndarray:
    h, w = plane.shape
    h_pad = int(np.ceil(h / 8) * 8) - h
    w_pad = int(np.ceil(w / 8) * 8) - w
    h_blocks = (h + h_pad) // 8
    w_blocks = (w + w_pad) // 8
    canvas = np.pad(plane, [(0, h_pad), (0, w_pad)], mode="edge")
    canvas = canvas.reshape(h_blocks, 8, w_blocks, 8)
    blocks = np.ascontiguousarray(canvas.transpose((0,2,1,3)))
    return blocks


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


def get_qtable(quality, chrominance=False):
    scale = jpeg_quality_scaling(quality)
    default = DEFAULT_QTABLE_C if chrominance else DEFAULT_QTABLE_Y
    tab = (default * scale + 50) / 100
    tab = np.clip(tab, a_min=1, a_max=255)
    return tab.astype(np.uint8)
