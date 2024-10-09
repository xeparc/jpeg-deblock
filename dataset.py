import collections
import collections.abc
import os
from collections import defaultdict
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import torchvision.transforms.functional

from utils import is_image
from jpegutils import JPEGTransforms, jpeg_quality_scaling


class DatasetQuantizedJPEG(torch.utils.data.Dataset):

    def __init__(
            self,
            image_dirs: Union[str, collections.abc.Iterable],
            region_size: int,
            patch_size: int,
            num_patches: int,
            min_quality: int,
            max_quality: int,
            target_quality: int,
            subsample: int,
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
            cached: bool = False,
            cache_memory: float = 16.0
    ):

        if isinstance(image_dirs, str):
            image_dirs = (image_dirs,)

        assert subsample in (444, 422, 420, 411, 440)
        assert 1 <= min_quality <= max_quality <= 100
        assert 1 <= target_quality <= 100
        assert 1 <= patch_size

        self.region_size = region_size
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
        self.cached = cached

        self.image_paths = []
        for _dir in image_dirs:
            assert os.path.exists(_dir)
            for root, _, filenames in os.walk(_dir):
                for fname in filter(is_image, filenames):
                    self.image_paths.append(os.path.join(root, fname))

        # Load images to memory if `cached` == True
        self.imgcache = {}
        if self.cached:
            max_cache_size = cache_memory * (2 ** 30)
            cur_cache_size = 0
            for path in self.image_paths:
                img = Image.open(path)
                img = torchvision.transforms.functional.pil_to_tensor(img)
                sz = img.numel()
                if cur_cache_size + sz < max_cache_size:
                    self.imgcache[path] = img
                    cur_cache_size += sz
                else:
                    break

        # Initialize crop transform. Crop patches from central region optionally
        if self.region_size >= self.patch_size:
            self.crop = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(self.region_size),
                torchvision.transforms.RandomCrop(self.patch_size)
            ])
        else:
            self.crop = torchvision.transforms.RandomCrop(self.patch_size)

        # Instead of sampling from uniform distribution
        # U[`min_quality`, `max_quality`], it makes more sense to sample from
        # a distribution that has heavier tail on lower qualities.
        # JPEG quality scaling is nonlinear for q < 50, and the change in
        # image degradation is way more severe for lower qualities.
        # That's why we'll sample with predefined probability for each
        # `quality` in [1, 100].
        # The sampling probability `p` will be proportional to the
        # quality scaling a.k.a quantization level.
        self.quality_a  = np.arange(min_quality, max_quality, dtype=np.float32)
        self.quality_p  = np.array([jpeg_quality_scaling(x) for x in self.quality_a], dtype=np.float32)
        self.quality_p /= np.sum(self.quality_p)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        # Get filepath of image
        impath = self.image_paths[index]

        # Read image
        if self.cached and impath in self.imgcache:
            image = self.imgcache[impath]
        else:
            image = torchvision.transforms.functional.pil_to_tensor(Image.open(impath))

        # Sample array of JPEG qualities
        qualities = np.random.choice(self.quality_a, self.num_patches, p=self.quality_p)

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
            patchT = JPEGTransforms(patch.permute(1,2,0).numpy())
            # Save metadata
            result["filepath"].append(impath)
            result["quality"].append(torch.tensor(quality, dtype=torch.float32).view(1))

            # High quality RGB
            if self.use_hq_rgb:
                result["hq_rgb"].append(transform_rgb(patchT.get_rgb()))
            # High quality YCbCr (downsampled + upsampled)
            if self.use_hq_ycc:
                y, cb, cr = patchT.get_ycc_planes(self.subsample)
                ycc = patchT.get_upsampled_ycc_planes(self.subsample)
                result["hq_y"].append(transform_ycc(y))
                result["hq_cb"].append(transform_ycc(cb))
                result["hq_cr"].append(transform_ycc(cr))
                result["hq_ycc"].append(transform_ycc(ycc))
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
                ycc = quantizedT.get_upsampled_ycc_planes(self.subsample)
                result["lq_y"].append(transform_ycc(y))
                result["lq_cb"].append(transform_ycc(cb))
                result["lq_cr"].append(transform_ycc(cr))
                result["lq_ycc"].append(transform_ycc(ycc))
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
    def collate_fn(batch):
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
                res[k] = torch.stack(collection, dim=0)
            else:
                res[k] = collection
        return res


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

        return patches.reshape(C, -1, h, w).transpose(0, 1)
