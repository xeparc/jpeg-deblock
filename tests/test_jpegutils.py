import os
from itertools import product
import numpy as np
import pytest
import torch
import torchvision

from PIL import Image
from turbojpeg import (
    TurboJPEG,
    TJCS_RGB,
    TJFLAG_ACCURATEDCT,
    TJSAMP_444,
    TJSAMP_440,
    TJSAMP_422,
    TJSAMP_411,
    TJSAMP_420
)

import context
from jpegutils import (
    JPEGTransforms,
    rgb2ycc,
    ycc2rgb,
    subsample_chrominance,
    upsample_chrominance,
    to_data_units,
    SUBSAMPLE_FACTORS
)
from utils import is_image


TEST_IMAGES_DIR = "data/live1/refimgs/"
OUTPUT_DIR = "data/tests-output/"
TEST_IMAGE_PATHS = [x.path for x in os.scandir(TEST_IMAGES_DIR) if is_image(x.name)]
TEST_IMAGES = [np.array(Image.open(path)) for path in TEST_IMAGE_PATHS]


@pytest.mark.parametrize("image_path", TEST_IMAGE_PATHS)
def test_rgb_ycc_rgb(image_path: str):

    # Read RGB
    rgb_true = np.array(Image.open(image_path))

    # Test uint conversion
    ycc = rgb2ycc(rgb_true)
    rgb = ycc2rgb(ycc)
    diff = np.abs(rgb.astype(np.int32) - rgb_true.astype(np.int32))
    assert np.max(diff) <= 1

    # Test float conversion
    x = rgb_true.astype(np.float32) / 255.0
    ycc_float = rgb2ycc(x)
    rgb_float = ycc2rgb(ycc_float)
    diff = np.abs(rgb_float - x)
    assert np.max(diff) < 1/255


@pytest.mark.parametrize("image, subsample",
                         list(product(TEST_IMAGES, SUBSAMPLE_FACTORS.keys())))
def test_subsampling(image, subsample):

    h_tol, w_tol = SUBSAMPLE_FACTORS[subsample]
    ycc = rgb2ycc(image)
    sub = subsample_chrominance(ycc, subsample)
    ups = upsample_chrominance(sub, subsample)
    for channel, plane in zip(np.moveaxis(ycc, 2, 0), ups):
        h, w = channel.shape
        uh, uw = plane.shape
        assert h <= uh and uh - h <= h_tol
        assert w <= uw and uw - w <= w_tol


@pytest.mark.parametrize("image, subsample",
                         list(product(TEST_IMAGES, SUBSAMPLE_FACTORS.keys())))
def test_to_data_units(image, subsample):

    ycc = rgb2ycc(image)
    ycc_sub = subsample_chrominance(ycc, subsample)
    data_units = to_data_units(ycc_sub)
    for du, plane in zip(data_units, ycc_sub):
        assert du.ndim == 4
        assert du.shape[-1] == 8
        assert du.shape[-2] == 8
        nh, nw = du.shape[:2]
        x = np.transpose(du, (0,2,1,3)).reshape(nh * 8, nw * 8)
        # Crop
        x = x[:plane.shape[0], :plane.shape[1]]
        assert np.all(x == plane)


@pytest.mark.parametrize("image_path, quality",
                         list(product(TEST_IMAGE_PATHS,
                                      (10,15,20,30,50,80),
                                      )))
def test_encoding(image_path, quality, subsample=444, saveimgs=False):

    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]

    if saveimgs:
        savedir = os.path.join(OUTPUT_DIR, "test_encoding/")
        os.makedirs(savedir, exist_ok=True)
        name = '.'.join(os.path.basename(image_path).split('.')[:-1])
        name1 = name + f"-quality={quality}-{subsample}-memquant.png"
        name2 = name + f"-quality={quality}-{subsample}-turbojpeg.jpg"

    tjsample = {
        444: TJSAMP_444,
        440: TJSAMP_440,
        422: TJSAMP_422,
        411: TJSAMP_411,
        420: TJSAMP_420
    }

    jpeg = JPEGTransforms(image)
    enc = jpeg.encode(quality, subsample)
    compressed = jpeg.decode_rgb(enc, subsample)

    # Save the image with in-memory quantization
    if saveimgs:
        torchvision.io.write_png(
            torch.from_numpy(compressed).permute(2,0,1),
            os.path.join(savedir, name1))

    turbo = TurboJPEG()
    buff = turbo.encode(image, quality=quality,
                        jpeg_subsample=tjsample[subsample],
                        pixel_format=TJCS_RGB, flags=TJFLAG_ACCURATEDCT)
    decoded = turbo.decode(buff, pixel_format=TJCS_RGB, flags=TJFLAG_ACCURATEDCT)

    # Save the image with libturbojpeg compression
    if saveimgs:
        with open(os.path.join(savedir, name2), mode="wb") as f:
            f.write(buff)

    # Check difference
    diff = np.abs(compressed.astype(np.int32) - decoded.astype(np.int32))
    diff_blocks = to_data_units([diff[:,:,0], diff[:,:,1], diff[:,:,2]])
    diff_blocks = np.array(diff_blocks)
    nblock_h, nblock_w = diff_blocks.shape[1:3]
    diff_blocks = np.array(diff_blocks).reshape(3, nblock_h, nblock_w, 64)
    diff_blocks = np.max(diff_blocks, axis=(0,3))
    x = np.count_nonzero(diff_blocks > 10)
    assert x / (nblock_h * nblock_w) <= 0.05
