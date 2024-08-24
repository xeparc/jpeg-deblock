import collections
import collections.abc
import math
import os
from collections import defaultdict
from typing import List, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from turbojpeg import TurboJPEG

from utils import is_image
from jpegutils import JPEGTransforms

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


class  ToDCTTensor:

    def __init__(self,
                 luma_mean=None, luma_std=None,
                 chroma_mean=None, chroma_std=None,
        ):
        if luma_mean is not None:
            assert luma_mean.shape == (8,8)
            self.luma_mean = torch.as_tensor(luma_mean).float()
        else:
            self.luma_mean = torch.zeros((8,8), dtype=torch.float32)

        if luma_std is not None:
            assert luma_std.shape == (8,8)
            self.luma_std = torch.as_tensor(luma_std).float()
        else:
            self.luma_std = torch.ones((8,8), dtype=torch.float32)

        if luma_mean is None and luma_std is None:
            self.skip_luma = True
        else:
            self.skip_luma = False

        if chroma_mean is not None:
            assert chroma_mean.shape == (8,8)
            self.chroma_mean = torch.as_tensor(chroma_mean).float()
        else:
            self.chroma_mean = torch.zeros((8,8), dtype=torch.float32)

        if chroma_std is not None:
            assert chroma_std.shape == (8,8)
            self.chroma_std = torch.as_tensor(chroma_std).float()
        else:
            self.chroma_std = torch.ones((8,8), dtype=torch.float32)

        if chroma_mean is None and chroma_std is None:
            self.skip_chroma = True
        else:
            self.skip_chroma = False

    def __call__(self, dct: np.ndarray, chroma: bool):
        assert dct.ndim == 4
        assert dct.shape[2] == dct.shape[3] == 8
        out_shape = dct.shape[:-2] + (64,)
        dct = torch.from_numpy(dct)
        if chroma:
            if self.skip_chroma:
                res = dct
            else:
                res = (dct - self.chroma_mean) / self.chroma_std
        else:
            if self.skip_luma:
                res = dct
            else:
                res = (dct - self.luma_mean) / self.luma_std
        return res.view(out_shape).permute(2,0,1).contiguous()


class ToQTTensor(nn.Module):

    def __init__(self, invert=False):
        self.invert = invert

    def __call__(self, qtable: np.ndarray):
        x = torch.as_tensor((qtable.astype(np.float32) - 1) / 254).ravel()
        if self.invert:
            return 1.0 - x
        else:
            return x


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


class DatasetQuantizedJPEG(torch.utils.data.Dataset):

    def __init__(
            self,
            image_dirs: Union[str, collections.abc.Iterable],
            patch_size: int,
            num_patches: int,
            min_quality: int,
            max_quality: int,
            target_quality: int,
            subsample: int,
            # normalize_rgb: bool,
            # normalize_ycc: bool,
            transform_dct: Callable,
            transform_qt: Callable,
            use_lq_rgb: bool,
            use_lq_ycc: bool,
            use_lq_dct: bool,
            use_hq_rgb: bool,
            use_hq_ycc: bool,
            use_hq_dct: bool,
            use_qt: bool,
            seed = None,
            device = "cpu",
            cached: bool = False
    ):

        if isinstance(image_dirs, str):
            image_dirs = (image_dirs,)

        assert subsample in (444, 422, 420, 411, 440)
        assert 1 <= min_quality <= max_quality <= 100
        assert 1 <= target_quality <= 100
        assert 1 <= patch_size

        self.patch_size = patch_size
        self.subsample = subsample
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.target_quality = target_quality
        self.num_patches = num_patches
        self.transform_rgb = torchvision.transforms.ToTensor()
        self.transform_ycc = torchvision.transforms.ToTensor()
        self.transform_dct = transform_dct
        self.transform_qt  = transform_qt
        self.use_lq_rgb = use_lq_rgb
        self.use_lq_ycc = use_lq_ycc
        self.use_lq_dct = use_lq_dct
        self.use_hq_rgb = use_hq_rgb
        self.use_hq_ycc = use_hq_ycc
        self.use_hq_dct = use_hq_dct
        self.use_qt = use_qt
        self.seed = seed
        self.device = torch.device(device)
        self.cached = cached

        self.image_paths = []
        for _dir in image_dirs:
            assert os.path.exists(_dir)
            for root, _, filenames in os.walk(_dir):
                for fname in filter(is_image, filenames):
                    self.image_paths.append(os.path.join(root, fname))

        self.crop = torchvision.transforms.RandomCrop(size=self.patch_size)
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        # Get filepath of image
        impath = self.image_paths[index]
        # Read image
        image = Image.open(impath)

        # Sample array of JPEG qualities
        qualities = self.rng.integers(
            self.min_quality, self.max_quality, size=self.num_patches)

        transform_rgb = self.transform_rgb
        transform_ycc = self.transform_ycc
        transform_dct = self.transform_dct
        transform_qt  = self.transform_qt
        result = defaultdict(list)

        # Extract patches
        for quality in qualities:
            quality = quality
            # Crop patch
            patch = self.crop(image)
            patchT = JPEGTransforms(np.array(patch))
            # Save metadata
            result["filepath"].append(impath)
            result["quality"].append(quality)

            # High quality RGB
            if self.use_hq_rgb:
                result["hq_rgb"].append(transform_rgb(patchT.get_rgb()))
            # High quality YCbCr (downsampled + upsampled)
            if self.use_hq_ycc:
                y, cb, cr = patchT.get_ycc_planes(self.subsample)
                result["hq_y"].append(transform_ycc(y))
                result["hq_cb"].append(transform_ycc(cb))
                result["hq_cr"].append(transform_ycc(cr))
            # High quality DCT coefficients
            if self.use_hq_dct:
                dct = patchT.get_dct_planes(self.subsample)
                result["hq_dct_y"].append(transform_dct(dct[0], chroma=False))
                result["hq_dct_cb"].append(transform_dct(dct[1], chroma=True))
                result["hq_dct_cr"].append(transform_dct(dct[2], chroma=True))

            quantized = patchT.encode(quality=quality, subsample=self.subsample)
            quantized_rgb = patchT.decode_rgb(quantized, subsample=self.subsample)
            quantizedT = JPEGTransforms(quantized_rgb)

            # Quantization tables
            if self.use_qt:
                y_qt = quantizedT.get_y_qtable(quality)
                c_qt = quantizedT.get_c_qtable(quality)
                result["qt_y"].append(transform_qt(y_qt))
                result["qt_c"].append(transform_qt(c_qt))

            # Low quality RGB
            if self.use_lq_rgb:
                result["lq_rgb"].append(transform_rgb(quantizedT.get_rgb()))
            # Low quality YCbCr (downsampled + upsampled)
            if self.use_lq_ycc:
                y, cb, cr = quantizedT.get_ycc_planes(self.subsample)
                result["lq_y"].append(transform_ycc(y))
                result["lq_cb"].append(transform_ycc(cb))
                result["lq_cr"].append(transform_ycc(cr))
            # Low quality DCT coefficients
            if self.use_lq_dct:
                dct = quantizedT.get_dct_planes(self.subsample)
                result["lq_dct_y"].append(transform_dct(dct[0], chroma=False))
                result["lq_dct_cb"].append(transform_dct(dct[1], chroma=True))
                result["lq_dct_cr"].append(transform_dct(dct[2], chroma=True))

        if self.num_patches == 1:
            return {k: v[0] for k,v in result.items()}
        else:
            return result

    @staticmethod
    def collate_fn(batch, device):
        temp = {}
        for item in batch:
            for k, v in item.items():
                if isinstance(v, list):
                    temp.setdefault(k, []).extend(v)
                else:
                    temp.setdefault(k, []).append(v)
        res = {}
        for k, collection in temp.items():
            if isinstance(collection[0], torch.Tensor):
                res[k] = torch.stack(collection, dim=0).to(device=device)
            else:
                res[k] = collection
        return res


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




if __name__ == "__main__":
    pass
