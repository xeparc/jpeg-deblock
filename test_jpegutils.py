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

from jpegutils import (
    JPEGTransforms,
    rgb2ycc,
    ycc2rgb,
    subsample_chrominance,
    upsample_chrominance,
    to_data_units,
    SUBSAMPLE_FACTORS
)


TEST_IMAGES_DIR = "data/Live1-Classic5/live1/refimgs/"
TEST_IMAGE_PATHS = [x.path for x in os.scandir(TEST_IMAGES_DIR)]
TEST_IMAGES = [np.array(Image.open(path)) for path in TEST_IMAGE_PATHS]


# def test_yuv2rgb_conversion(save=False):

#     MAXDELTA = 1
#     transform = YCbCr2RGB()

#     failed = []
#     deltas = []
    
#     for item in os.scandir(TEST_IMAGES_DIR):
#         extension = item.name.split('.')[-1]
#         if extension not in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
#             continue
#         image = Image.open(item.path)
#         rgb_true = np.array(image)

#         dat = JPEGImageData(rgb_true, subsample="444")

#         yuv = dat.get_upsampled_yuv() / 255.0
#         rgb = transform(yuv.unsqueeze(0))[0]
#         rgb = torch.trunc(rgb * 255).to(dtype=torch.int)

#         if save:
#             os.makedirs("tests/yuv2rgb", exist_ok=True)
#             basename = '.'.join(item.name.split('.')[:-1])
#             savepath = os.path.join("tests/yuv2rgb/", basename + ".jpg")
#             torchvision.io.write_jpeg(rgb, savepath, quality=100)

#         dt = torch.max(torch.abs(rgb - dat.rgb.int()).view(3, -1), dim=1).values
#         if torch.any(dt > MAXDELTA):
#             print("f", end='', flush=True)
#             failed.append(item.path)
#             deltas.append(tuple(dt.numpy().ravel()))
#         else:
#             print(".", end='', flush=True)

#     if not failed:
#         print("\nOK\n")
#     else:
#         print("\n\ntest_tuv2rgb_conversion() failed for:\n")
#         for delta, path in zip(deltas, failed):
#             name = os.path.basename(path)
#             print(f"\t{name:<30}\tRGB deltas = {delta}", flush=True)


# def test_rgb2yuv_conversion():

#     MAXDELTA = 1
#     images_dir = "data/Live1-Classic5/live1/refimgs/"

#     transform = RGB2YCbCr()

#     failed = []
#     deltas = []
    
#     for item in os.scandir(images_dir):
#         extension = item.name.split('.')[-1]
#         if extension not in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
#             continue
#         image = Image.open(item.path)
#         rgb_true = np.array(image)

#         dat = JPEGImageData(rgb_true, subsample="444")
#         yuv_true = dat.get_upsampled_yuv().to(dtype=torch.int16)

#         rgb = (dat.rgb / 255.0).float()
#         yuv = transform(rgb.unsqueeze(0))[0]
#         yuv = torch.round(yuv * 255).to(dtype=torch.int16)

#         deltas_Y = torch.abs(yuv[0] - yuv_true[0])
#         deltas_U = torch.abs(yuv[1] - yuv_true[1])
#         deltas_V = torch.abs(yuv[2] - yuv_true[2])

#         maxdelta_Y = torch.max(deltas_Y.ravel()).item()
#         maxdelta_U = torch.max(deltas_U.ravel()).item()
#         maxdelta_V = torch.max(deltas_V.ravel()).item()

#         if not (max(maxdelta_Y, maxdelta_U, maxdelta_V) <= MAXDELTA):
#             print("f", end='', flush=True)
#             failed.append(item.path)
#             deltas.append((maxdelta_Y, maxdelta_U, maxdelta_V))
#         else:
#             print(".", end='', flush=True)

#     if not failed:
#         print("\nOK\n")
#     else:
#         print("\n\ntest_rgb2yuv_conversion() failed for:\n")
#         for delta, path in zip(deltas, failed):
#             name = os.path.basename(path)
#             print(f"\t{name:<30}\tYUV deltas = {delta}", flush=True)


# def test_rgb_yuv_rgb_conversion():

#     MAXDELTA = 1
#     images_dir = "data/Live1-Classic5/live1/refimgs/"

#     toyuv = RGB2YCbCr()
#     torgb = YCbCr2RGB()

#     failed = []
#     deltas = []

#     for item in os.scandir(images_dir):
#         extension = item.name.split('.')[-1]
#         if extension not in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
#             continue
#         image = Image.open(item.path)
#         rgb_true = np.array(image)

#         x = torch.from_numpy(rgb_true).permute(2,0,1) / 255.0
#         yuv = toyuv(x.unsqueeze(0))
#         rgb = torch.round(torgb(yuv)[0] * 255).int()

#         dt = torch.max(torch.abs(rgb - (255 * x).int()), dim=0).values

#         if torch.any(dt > MAXDELTA):
#             print("f", end='', flush=True)
#             failed.append(item.path)
#             deltas.append(tuple(dt))
#         else:
#             print(".", end='', flush=True)

#     if not failed:
#         print("\nOK\n")
#     else:
#         print("\n\ntest_rgb_yuv_rgb_conversion() failed for:\n")
#         for delta, path in zip(deltas, failed):
#             name = os.path.basename(path)
#             print(f"\t{name:<30}\RGB deltas = {delta}", flush=True)


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
def test_encoding(image_path, quality, subsample="444", saveimgs=True):

    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]

    if saveimgs:
        savedir = "tests/encoding/"
        os.makedirs(savedir, exist_ok=True)
        name = '.'.join(os.path.basename(image_path).split('.')[:-1])
        name1 = name + f"-quality={quality}-{subsample}-memquant.png"
        name2 = name + f"-quality={quality}-{subsample}-turbojpeg.jpg"

    tjsample = {
        "444": TJSAMP_444,
        "440": TJSAMP_440,
        "422": TJSAMP_422,
        "411": TJSAMP_411,
        "420": TJSAMP_420
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
    x = np.count_nonzero(diff_blocks > 5)
    assert x / (nblock_h * nblock_w) <= 0.05

# if __name__ == "__main__":
#     # test_yuv2rgb_conversion(save=False)
#     # test_rgb2yuv_conversion()
#     # test_rgb_yuv_rgb_conversion()
#     # test_quantization()