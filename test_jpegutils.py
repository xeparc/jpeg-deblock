import numpy as np
import os
import torch
import torchvision

from PIL import Image
from jpegutils import YCbCr2RGB, RGB2YCbCr, JPEGImageData


TEST_IMAGES_DIR = "data/Live1-Classic5/live1/refimgs/"


def test_yuv2rgb_conversion(save=False):

    MAXDELTA = 1
    transform = YCbCr2RGB()

    failed = []
    deltas = []
    
    for item in os.scandir(TEST_IMAGES_DIR):
        extension = item.name.split('.')[-1]
        if extension not in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
            continue
        image = Image.open(item.path)
        rgb_true = np.array(image)

        dat = JPEGImageData(rgb_true, subsample="444")

        yuv = dat.get_upsampled_yuv() / 255.0
        rgb = transform(yuv.unsqueeze(0))[0]
        rgb = torch.trunc(rgb * 255).to(dtype=torch.int)

        if save:
            os.makedirs("tests/yuv2rgb", exist_ok=True)
            basename = '.'.join(item.name.split('.')[:-1])
            savepath = os.path.join("tests/yuv2rgb/", basename + ".jpg")
            torchvision.io.write_jpeg(rgb, savepath, quality=100)

        dt = torch.max(torch.abs(rgb - dat.rgb.int()).view(3, -1), dim=1).values
        if torch.any(dt > MAXDELTA):
            print("f", end='', flush=True)
            failed.append(item.path)
            deltas.append(tuple(dt.numpy().ravel()))
        else:
            print(".", end='', flush=True)

    if not failed:
        print("\nOK\n")
    else:
        print("\n\ntest_tuv2rgb_conversion() failed for:\n")
        for delta, path in zip(deltas, failed):
            name = os.path.basename(path)
            print(f"\t{name:<30}\tRGB deltas = {delta}", flush=True)


def test_rgb2yuv_conversion():

    MAXDELTA = 1
    images_dir = "data/Live1-Classic5/live1/refimgs/"

    transform = RGB2YCbCr()

    failed = []
    deltas = []
    
    for item in os.scandir(images_dir):
        extension = item.name.split('.')[-1]
        if extension not in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
            continue
        image = Image.open(item.path)
        rgb_true = np.array(image)

        dat = JPEGImageData(rgb_true, subsample="444")
        yuv_true = dat.get_upsampled_yuv().to(dtype=torch.int16)

        rgb = (dat.rgb / 255.0).float()
        yuv = transform(rgb.unsqueeze(0))[0]
        yuv = torch.round(yuv * 255).to(dtype=torch.int16)

        deltas_Y = torch.abs(yuv[0] - yuv_true[0])
        deltas_U = torch.abs(yuv[1] - yuv_true[1])
        deltas_V = torch.abs(yuv[2] - yuv_true[2])

        maxdelta_Y = torch.max(deltas_Y.ravel()).item()
        maxdelta_U = torch.max(deltas_U.ravel()).item()
        maxdelta_V = torch.max(deltas_V.ravel()).item()

        if not (max(maxdelta_Y, maxdelta_U, maxdelta_V) <= MAXDELTA):
            print("f", end='', flush=True)
            failed.append(item.path)
            deltas.append((maxdelta_Y, maxdelta_U, maxdelta_V))
        else:
            print(".", end='', flush=True)

    if not failed:
        print("\nOK\n")
    else:
        print("\n\ntest_rgb2yuv_conversion() failed for:\n")
        for delta, path in zip(deltas, failed):
            name = os.path.basename(path)
            print(f"\t{name:<30}\tYUV deltas = {delta}", flush=True)


def test_rgb_yuv_rgb_conversion():

    MAXDELTA = 1
    images_dir = "data/Live1-Classic5/live1/refimgs/"

    toyuv = RGB2YCbCr()
    torgb = YCbCr2RGB()

    failed = []
    deltas = []

    for item in os.scandir(images_dir):
        extension = item.name.split('.')[-1]
        if extension not in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
            continue
        image = Image.open(item.path)
        rgb_true = np.array(image)

        x = torch.from_numpy(rgb_true).permute(2,0,1) / 255.0
        yuv = toyuv(x.unsqueeze(0))
        rgb = torch.round(torgb(yuv)[0] * 255).int()

        dt = torch.max(torch.abs(rgb - (255 * x).int()), dim=0).values

        if torch.any(dt > MAXDELTA):
            print("f", end='', flush=True)
            failed.append(item.path)
            deltas.append(tuple(dt))
        else:
            print(".", end='', flush=True)

    if not failed:
        print("\nOK\n")
    else:
        print("\n\ntest_rgb_yuv_rgb_conversion() failed for:\n")
        for delta, path in zip(deltas, failed):
            name = os.path.basename(path)
            print(f"\t{name:<30}\RGB deltas = {delta}", flush=True)


def test_quantization(subsample="444"):
    from turbojpeg import TurboJPEG, TJCS_RGB, TJFLAG_ACCURATEDCT, TJSAMP_444, TJSAMP_422

    output_dir = "tests/quantization/"
    os.makedirs(output_dir, exist_ok=True)
    turbo = TurboJPEG()

    for item in os.scandir(TEST_IMAGES_DIR):
        extension = item.name.split('.')[-1]
        if extension not in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
            continue

        basename = '.'.join(item.name.split('.')[:-1])
        image = np.array(Image.open(item.path))
        dat = JPEGImageData(image, subsample=subsample)

        for q in (10, 15, 20, 25, 30, 40, 50, 80):
            filename = basename + f"-quality={q}-turbojpeg.jpg"
            savepath = os.path.join(output_dir, filename)
            # Encode with libturbojpeg
            with open(savepath, mode="wb") as f:
                buff = turbo.encode(image, q, pixel_format=TJCS_RGB,
                            jpeg_subsample=TJSAMP_422, flags=TJFLAG_ACCURATEDCT)
                f.write(buff)
            # Encode in-memory
            quantized = dat.quantize(q)
            filename = basename + f"-quality={q}-memquant.png"
            savepath = os.path.join(output_dir, filename)
            with open(savepath, mode="wb") as f:
                torchvision.io.write_png(quantized.rgb, savepath)
        print(item.name)


if __name__ == "__main__":
    # test_yuv2rgb_conversion(save=False)
    # test_rgb2yuv_conversion()
    # test_rgb_yuv_rgb_conversion()
    test_quantization()