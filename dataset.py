import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from turbojpeg import TurboJPEG
from jpeglib import read_dct, read_spatial

from utils import RunningMeanStd
from jpegutils import get_jpeg_data, JPEGData


IMAGES_PATH = "data/BSDS500/BSDS500/data/images/"
OUTPUT_PATH = "data/"
QUALITY = [10, 20, 30, 40, 50, 60, 80]
DEBUG = 0


class ExtractSubpatches:
    """ Extracts multiple non-overlapping subpatches from a source image."""

    def __init__(self, height, width, pad=False):
        self.subpatch_h = int(height)
        self.subpatch_w = int(width)
        self.padding = bool(pad)

    def __call__(self, image):
        C, H, W = image.shape
        h, w = self.subpatch_h, self.subpatch_w
        if self.padding:
            pad_w = W % w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_h = H % h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            padded_image = nn.functional.pad(
                image, (pad_left, pad_right, pad_top, pad_bottom))
        else:
            newH = (H // h) * h
            newW = (W // w) * w
            padded_image = image[:, :newH, :newW]

        # (C, H, W)
        patches = padded_image.unfold(dimension=1, size=h, step=h)
        # (C, nH, W, height)
        patches = patches.unfold(dimension=2, size=w, step=w)
        # (C, nH, nW, height, width)

        # Test if they are the same
        if DEBUG:
            patches_orig = patches.permute(0, 1, 3, 2, 4).contiguous()
            assert torch.all(patches_orig.reshape(image.shape) == image)

        return patches.reshape(C, -1, h, w).transpose(0, 1)


class AddCheckerboardChannel:
    """Adds a channel with 8x8 checkerboard pattern to image. This simulates
    the neighbouring DCT blocks."""

    def __init__(self, gain=0.5, offset_h=0, offset_w=0):
        assert offset_h < 8
        assert offset_w < 8
        self.gain = float(gain)
        self.offset_h = int(offset_h)
        self.offset_w = int(offset_w)

    def __call__(self, image):
        C, H, W = image.shape
        h, w = math.ceil(H / 8), math.ceil(W / 8)
        grid = torch.ones((h, w), dtype=torch.float32)
        grid[::2, ::2] = -grid[::2, ::2]
        block = torch.full((8,8), self.gain, dtype=torch.float32)
        channel = torch.kron(grid, block)
        channel = torch.roll(channel, shifts=(self.offset_h, self.offset_w), dims=(0,1))
        channel = channel[None, :H, :W]
        return torch.concat([image, channel], dim=0)


def cutp(image, height, width, pad=False):
    """Cuts `image` into nonoverlapping patches."""
    C, H, W = image.shape
    if pad:
        pad_w = W % width
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = H % height
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded_image = nn.functional.pad(
            image, (pad_left, pad_right, pad_top, pad_bottom))
    else:
        h = (H // height) * height
        w = (W // width)  * width
        padded_image = image[:, :h, :w]

    # (C, H, W)
    patches = padded_image.unfold(dimension=1, size=height, step=height)
    # (C, nH, W, height)
    patches = patches.unfold(dimension=2, size=width, step=width)
    # (C, nH, nW, height, width)

    # Test if they are the same
    if DEBUG:
        patches_orig = patches.permute(0, 1, 3, 2, 4).contiguous()
        assert torch.all(patches_orig.reshape(image.shape) == image)

    return patches.reshape(C, -1, height, width).transpose(0, 1)



# A custom Dataset class must implement three functions:
#       __init__, __len__, and __getitem__.
class DatasetDeblockPatches(torch.utils.data.Dataset):

    def __init__(self, compressed_dir, originals_dir, subpatch_size=40,
                 normalize=False, checkerboard=False, device="cpu"):
        super().__init__()
        assert os.path.exists(compressed_dir)
        assert os.path.exists(originals_dir)

        self.compressed_dir = os.path.normpath(compressed_dir)
        self.originals_dir  = os.path.normpath(originals_dir)
        self.subpatch_size = subpatch_size
        self.tofloat = torchvision.transforms.ConvertImageDtype(torch.float32)
        self.device = torch.device(device)

        input_transforms = [self.tofloat]
        target_transforms = [self.tofloat]
        if normalize:
            norm = torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            input_transforms.append(norm)
            target_transforms.append(norm)
        if checkerboard:
            input_transforms.append(AddCheckerboardChannel())
        self.input_transform = torchvision.transforms.Compose(input_transforms)
        self.target_transform = torchvision.transforms.Compose(target_transforms)

        self.razor = ExtractSubpatches(subpatch_size, subpatch_size, pad=False)

        # Load images
        self.inputs = []
        self.targets = []
        for item in os.scandir(self.compressed_dir):
            if item.name.lower().endswith(".jpg") or item.name.lower().endswith(".jpeg"):
                # Find the pairing target image in `originals_dir` for `item`
                target_path = os.path.join(self.originals_dir, item.name)
                # Load images from disk
                trainimg = torchvision.io.read_image(item.path)
                targetimg = torchvision.io.read_image(target_path)
                # Apply transform
                trainimg  = self.input_transform(trainimg)
                targetimg = self.target_transform(targetimg)
                trainpatches = self.razor(trainimg)
                targetpatches = self.razor(targetimg)
                self.inputs.extend(trainpatches)
                self.targets.extend(targetpatches)


    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        image, target = self.inputs[idx], self.targets[idx]
        # Convert to float and move to `self.device`
        x = image.to(self.device)
        y = target.to(self.device)
        return x, y


def encode_and_save_jpegs(input_dir, output_dir, quality=(80, 60, 40, 30, 20, 10)):
    """
    Re-encodes every JPEG image in `input_dir` (recursively) with quality Q and
    saves it in `output_dir`/Q/. The directory structure below `input_dir` is
    mirrored under `output_dir`/Q/

    Parameters:
    -----------
        input_dir: str
            The source directory from which JPEG images are read.
        output_dir: str
            The output directory in which re-encoded JPEG images are saved.
            If it does not exists, it's created.
        quality: iterable
            Quality settings passed to TurboJPEG encoder. For each Q in
            `quality`, a directory is created in `output_dir`. The directory
            structure if `input_dir`/ is mirrored in `output_dir`/Q/.

    Returns:
    --------
        None
    """

    # Create `output_dir` if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    turbo = TurboJPEG()

    # Encode loop
    for q in quality:
        savedir = os.path.join(output_dir, str(q))
        os.makedirs(savedir, exist_ok=True)

        for rootdir, _, filenames in os.walk(input_dir):
            prefix = os.path.commonpath([rootdir, input_dir])
            subpath = os.path.join(savedir, rootdir[len(prefix)+1:])
            os.makedirs(subpath, exist_ok=True)
            for file in filenames:
                if not file.endswith(".jpg") and not file.endswith(".jpeg"):
                    continue
                with open(os.path.join(rootdir, file), mode="rb") as f:
                    data = f.read()
                    img = turbo.decode(data)
                # Save
                encoded = turbo.encode(img, quality=q)
                savepath = os.path.join(subpath, file)
                with open(savepath, mode="wb") as f:
                    f.write(encoded)



def calculate_dct_mean_and_std(input_dir, verbose=True):

    estimator_Y = RunningMeanStd()
    estimator_C = RunningMeanStd()

    for rootdir, _, filenames in os.walk(input_dir):
        for file in filenames:
            fullpath = os.path.join(rootdir, file)
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                dctobj = read_dct(fullpath)
                dctY   = dctobj.Y  * dctobj.qt[0]
                dctCb  = dctobj.Cb * dctobj.qt[1]
                dctCr  = dctobj.Cr * dctobj.qt[1]
            if file.endswith(".png"):
                img = torchvision.io.read_image(fullpath)
                dat = get_jpeg_data(img.permute(1,2,0).numpy(), quality=100)
                dctY, dctCb, dctCr = dat.dctY.numpy(), dat.dctCb.numpy(), dat.dctCr.numpy()
            else:
                continue

            dctC = np.stack([dctCb, dctCr], axis=0)
            assert dctC.shape[0] == 2
            assert dctC.shape[3] == 8
            assert dctC.shape[4] == 8

            for block in dctY.reshape(-1, 8, 8):
                estimator_Y.update(block)

            for block in dctC.reshape(-1, 8, 8):
                estimator_C.update(block)
            
            if verbose:
                print(fullpath)

    return {
        "dct_Y_mean": estimator_Y.mean,
        "dct_C_mean": estimator_C.mean,
        "dct_Y_std":  estimator_Y.std,
        "dct_C_std":  estimator_C.std
    }


if __name__ == "__main__":

    # Calculate DCT coefficients mean and std
    images_paths = ["data/DIV2K/"]
    stats = calculate_dct_mean_and_std(images_paths[0])

    with open("DIV2K-DCT-coeff-stats.json", mode="wt") as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f)
    
