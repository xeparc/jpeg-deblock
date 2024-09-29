import itertools
import json
import os

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image

import context
from models.transforms import (
    ToDCTTensor,
    InverseDCT,
    ConvertYccToRGB,
)
from jpegutils import JPEGTransforms, rgb2ycc
from utils import is_image


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DCT_STATS_FILEPATH = "data/DIV2K-DCT-coeff-stats.json"
TEST_IMAGES_DIR = "data/live1/refimgs/"
TEST_IMAGES_PATHS = [x.path for x in os.scandir(TEST_IMAGES_DIR) if is_image(x.name)]


@pytest.mark.parametrize("filepath, normalize",
                         list(itertools.product(TEST_IMAGES_PATHS, (True, False))))
def test_InverseDCT(filepath, normalize):

    # Read DCT coefficient stats
    if normalize:
        with open(DCT_STATS_FILEPATH, mode="r") as f:
            stats = json.load(f)
        luma_mean =     torch.as_tensor(stats["dct_Y_mean"])
        luma_std =      torch.as_tensor(stats["dct_Y_std"])
        chroma_mean =   torch.as_tensor(stats["dct_C_mean"])
        chroma_std =    torch.as_tensor(stats["dct_C_std"])
    else:
        luma_mean = None
        luma_std  = None
        chroma_mean = None
        chroma_std = None

    # Read image
    rgb = torchvision.transforms.PILToTensor()(Image.open(filepath))
    ycc = rgb2ycc(rgb.permute(1,2,0).numpy())
    jpegT = JPEGTransforms(rgb.permute(1,2,0).numpy())
    h, w = jpegT.height, jpegT.width
    dct = jpegT.get_dct_planes(subsample=444)

    # Initialize transforms
    to_dct_tensor = ToDCTTensor(luma_mean, luma_std, chroma_mean, chroma_std)
    to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)
    to_rgb = ConvertYccToRGB()
    idct = InverseDCT(luma_mean, luma_std, chroma_mean, chroma_std)

    dct_y_10 = to_dct_tensor(dct[0], chroma=False)
    dct_cb_10 = to_dct_tensor(dct[1], chroma=True)
    dct_cr_10 = to_dct_tensor(dct[2], chroma=True)

    assert dct_y_10.ndim == 3 and dct_y_10.shape[0] == 64
    assert dct_cb_10.ndim == 3 and dct_cb_10.shape[0] == 64

    # Add batch dimension
    dct_y_10 = dct_y_10.unsqueeze(0)
    dct_cb_10 = dct_cb_10.unsqueeze(0)
    dct_cr_10 = dct_cr_10.unsqueeze(0)

    # Convert and test Y Plane
    converted_y = idct(dct_y_10, chroma=False)
    converted_y = torch.clamp(converted_y, 0.0, 1.0)
    converted_y = to_uint8(converted_y).squeeze(dim=0).squeeze(dim=0)
    assert converted_y.ndim == 2
    converted_y = converted_y[:h, :w].numpy()
    true_y = ycc[:,:,0]
    assert np.all(converted_y == true_y)

    # Convert and test Cb Plane
    converted_cb = idct(dct_cb_10, chroma=True)
    converted_cb = torch.clamp(converted_cb, 0.0, 1.0)
    converted_cb = to_uint8(converted_cb).squeeze(dim=0).squeeze(dim=0)
    assert converted_cb.ndim == 2
    converted_cb = converted_cb[:h, :w].numpy()
    true_cb = ycc[:,:,1]
    assert np.all(converted_cb == true_cb)

    # Convert and test Cr Plane
    converted_cr = idct(dct_cr_10, chroma=True)
    converted_cr = torch.clamp(converted_cr, 0.0, 1.0)
    converted_cr = to_uint8(converted_cr).squeeze(dim=0).squeeze(dim=0)
    assert converted_cr.ndim == 2
    converted_cr = converted_cr[:h, :w].numpy()
    true_cr = ycc[:,:,2]
    assert np.all(converted_cr == true_cr)


@pytest.mark.parametrize("filepath", TEST_IMAGES_PATHS)
def test_ConvertYccToRGB(filepath):

    # Read image
    rgb = torchvision.transforms.PILToTensor()(Image.open(filepath))
    ycc = rgb2ycc(rgb.permute(1,2,0).numpy())
    jpegT = JPEGTransforms(rgb.permute(1,2,0).numpy())
    h, w = jpegT.height, jpegT.width
    dct = jpegT.get_dct_planes(subsample=444)

    # Initialize transforms
    to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)
    to_rgb = ConvertYccToRGB()

    ycc10 = torchvision.transforms.ToTensor()(ycc)
    rgb10 = to_uint8(to_rgb(ycc10).clip(0.0, 1.0))
    assert torch.all(torch.abs(rgb10.int() - rgb.int()) <= 2)
