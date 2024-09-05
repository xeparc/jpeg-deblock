import json
import os

import numpy as np
import pytest
import torch
import torchvision

from dataset import DatasetQuantizedJPEG
from models.models import ToDCTTensor, ToQTTensor
from jpegutils import (
    JPEGTransforms,
    upsample_chrominance,
    ycc2rgb,
)
from models.models import InverseDCT


TEST_DATASET_DIR = "data/Live1-Classic5/live1/refimgs/"
LEN_DATASET = 29
DCT_STATS_FILEPATH = "data/DIV2K-DCT-coeff-stats.json"
DATAPOINTS_OUTPUT_DIR = "tests/dataset/"

os.makedirs(DATAPOINTS_OUTPUT_DIR, exist_ok=True)


class TestQuantizedDataset:

    def init(self):
        self.images_path = TEST_DATASET_DIR

        with open(DCT_STATS_FILEPATH, mode="r") as f:
            stats = json.load(f)
            self.transform_dct1 = ToDCTTensor(
                luma_mean=  np.asarray(stats["dct_Y_mean"]),
                luma_std=   np.asarray(stats["dct_Y_std"]),
                chroma_mean=np.asarray(stats["dct_C_mean"]),
                chroma_std= np.asarray(stats["dct_C_std"])
            )

        self.transform_dct2 = ToDCTTensor()

        self.transform_qt = ToQTTensor(invert=False)

        self.default_kwargs = dict(
            image_dirs = self.images_path,
            patch_size = 64,
            subsample = 420,
            min_quality = 10,
            max_quality = 80,
            target_quality = 100,
            num_patches = 1,
            transform_dct = self.transform_dct2,
            transform_qt = self.transform_qt,
            use_lq_rgb = False,
            use_lq_ycc = False,
            use_lq_dct = True,
            use_hq_rgb = True,
            use_hq_ycc = True,
            use_hq_dct = False,
            use_qt = True
        )
        self.disable_use_keys = dict(
            use_lq_rgb  = False,
            use_lq_ycc  = False,
            use_lq_dct  = False,
            use_hq_rgb  = False,
            use_hq_ycc  = False,
            use_hq_dct  = False,
            use_qt = False
        )
        self.default_keys = [
            "filepath",
            "quality",
            "lq_dct_y", "lq_dct_cb", "lq_dct_cr",
            "hq_rgb",
            "hq_y", "hq_cb", "hq_cr",
            "qt_y", "qt_c"
        ]

    def test_dataset_len(self):
        self.init()
        dataset = DatasetQuantizedJPEG(**self.default_kwargs)
        assert len(dataset) == LEN_DATASET

    def test_single_patch(self):
        self.init()
        self.default_kwargs.update(patch_size=1)
        dataset = DatasetQuantizedJPEG(**self.default_kwargs)
        for i in range(LEN_DATASET):
            res = dataset[i]
            assert isinstance(res, dict)

    def test_batch_of_patches(self):
        self.init()
        self.default_kwargs.update(num_patches=32)
        dataset = DatasetQuantizedJPEG(**self.default_kwargs)
        for i in range(LEN_DATASET):
            res = dataset[i]
            for key in self.default_keys:
                assert key in res
                assert len(res[key]) == 32

    def test_use_none(self):
        self.init()
        kwargs = self.default_kwargs
        kwargs.update(self.disable_use_keys)
        dataset = DatasetQuantizedJPEG(**kwargs)

        point = dataset[0]
        assert set(point.keys()) == {"filepath", "quality"}

    def test_use_rgb(self):
        self.init()
        self.default_kwargs.update(self.disable_use_keys)
        self.default_kwargs["use_hq_rgb"] = True
        self.default_kwargs["use_lq_rgb"] = True
        dataset = DatasetQuantizedJPEG(**self.default_kwargs)

        point = dataset[0]
        assert set(point.keys()) == {"filepath", "quality", "hq_rgb", "lq_rgb"}
        assert isinstance(point["hq_rgb"], torch.FloatTensor)
        assert point["hq_rgb"].shape == point["lq_rgb"].shape
        assert point["hq_rgb"].shape[0] == 3

    def test_use_ycc(self):
        self.init()
        self.default_kwargs.update(self.disable_use_keys)
        self.default_kwargs["use_hq_ycc"] = True
        self.default_kwargs["use_lq_ycc"] = True
        dataset = DatasetQuantizedJPEG(**self.default_kwargs)

        point = dataset[0]
        assert set(point.keys()) == {
            "filepath", "quality", "hq_y", "lq_y", "hq_cb", "lq_cb", "hq_cr", "lq_cr"}
        assert isinstance(point["hq_y" ], torch.FloatTensor)
        assert isinstance(point["hq_cb"], torch.FloatTensor)
        assert isinstance(point["hq_cr"], torch.FloatTensor)
        assert isinstance(point["lq_y" ], torch.FloatTensor)
        assert isinstance(point["lq_cb"], torch.FloatTensor)
        assert isinstance(point["lq_cr"], torch.FloatTensor)

        assert point["hq_y" ].ndim == 3
        assert point["hq_cb"].ndim == 3
        assert point["hq_cr"].ndim == 3
        assert point["lq_y" ].ndim == 3
        assert point["lq_cb"].ndim == 3
        assert point["lq_cr"].ndim == 3

    def test_use_dct(self):
        self.init()
        self.default_kwargs.update(self.disable_use_keys)
        self.default_kwargs["use_hq_dct"] = True
        self.default_kwargs["use_lq_dct"] = True
        dataset = DatasetQuantizedJPEG(**self.default_kwargs)

        point = dataset[0]
        assert set(point.keys()) == {"filepath", "quality",
                                     "hq_dct_y", "hq_dct_cb", "hq_dct_cr",
                                     "lq_dct_y", "lq_dct_cb", "lq_dct_cr"}

        assert isinstance(point["hq_dct_y" ], torch.FloatTensor)
        assert isinstance(point["hq_dct_cb"], torch.FloatTensor)
        assert isinstance(point["hq_dct_cr"], torch.FloatTensor)
        assert isinstance(point["lq_dct_y" ], torch.FloatTensor)
        assert isinstance(point["lq_dct_cb"], torch.FloatTensor)
        assert isinstance(point["lq_dct_cr"], torch.FloatTensor)

        assert point["hq_dct_y"].shape[0] == 64
        assert point["hq_dct_cb"].shape[0] == 64
        assert point["hq_dct_cr"].shape[0] == 64

    @pytest.mark.parametrize("invert", (True, False))
    def test_use_qt(self, invert):
        self.init()
        self.default_kwargs.update(self.disable_use_keys)
        self.default_kwargs["use_qt"] = True
        self.default_kwargs["transform_qt"] = ToQTTensor(invert)
        dataset = DatasetQuantizedJPEG(**self.default_kwargs)
        point = dataset[0]

        assert "qt_y" in point and "qt_c" in point
        assert point["qt_y"].dtype == torch.float32
        assert point["qt_c"].dtype == torch.float32
        assert 0 <= point["qt_c"].min() <= point["qt_c"].max() <= 1.0


    def test_save_datapoint(self):
        self.init()
        use_all = {k: not v for k,v in self.disable_use_keys.items()}
        self.default_kwargs.update(use_all)
        self.default_kwargs.update(max_quality=30)
        self.default_kwargs.update(patch_size=256)

        dataset = DatasetQuantizedJPEG(**self.default_kwargs)
        touint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)
        for i in range(LEN_DATASET):
            point = dataset[i]
            name = os.path.basename(point["filepath"])

            jpegT = JPEGTransforms(touint8(point["hq_rgb"]).permute(1,2,0).numpy())

            # Save HQ RGB image
            torchvision.io.write_png(
                touint8(point["hq_rgb"]),
                os.path.join(DATAPOINTS_OUTPUT_DIR, name + "-hq_rgb.png")
            )

            # Save LQ RGB image
            torchvision.io.write_png(
                touint8(point["lq_rgb"]),
                os.path.join(DATAPOINTS_OUTPUT_DIR, name + "-lq_rgb.png")
            )

            # Save an image from HQ YCbCr planes
            #
            y = touint8(point["hq_y"]).squeeze(0).numpy()
            cb = touint8(point["hq_cb"]).squeeze(0).numpy()
            cr = touint8(point["hq_cr"]).squeeze(0).numpy()


            ycc = upsample_chrominance([y, cb, cr], self.default_kwargs["subsample"])
            rgb = ycc2rgb(np.asarray(ycc).transpose(1,2,0))
            torchvision.io.write_png(
                torch.from_numpy(rgb).permute(2,0,1),
                os.path.join(DATAPOINTS_OUTPUT_DIR, name + "-hq_ycc.png")
            )

            # Save an image from LQ YCbCr planes
            #
            y = touint8(point["lq_y"]).squeeze(0).numpy()
            cb = touint8(point["lq_cb"]).squeeze(0).numpy()
            cr = touint8(point["lq_cr"]).squeeze(0).numpy()

            ycc = upsample_chrominance([y, cb, cr], self.default_kwargs["subsample"])
            rgb = ycc2rgb(np.asarray(ycc).transpose(1,2,0))
            torchvision.io.write_png(
                torch.from_numpy(rgb).permute(2,0,1),
                os.path.join(DATAPOINTS_OUTPUT_DIR, name + "-lq_ycc.png")
            )

            # Save an image from HQ DCT coeffs
            #
            hq_dct_y = point["hq_dct_y"]
            hq_dct_cb = point["hq_dct_cb"]
            hq_dct_cr = point["hq_dct_cr"]

            hq_dct_y = hq_dct_y.permute(1,2,0)
            hq_dct_cb = hq_dct_cb.permute(1,2,0)
            hq_dct_cr = hq_dct_cr.permute(1,2,0)

            hq_dct_y = hq_dct_y.reshape(hq_dct_y.shape[:-1] + (8,8))
            hq_dct_cb = hq_dct_cb.reshape(hq_dct_cb.shape[:-1] + (8,8))
            hq_dct_cr = hq_dct_cr.reshape(hq_dct_cr.shape[:-1] + (8,8))

            hq_dct = [hq_dct_y.numpy(), hq_dct_cb.numpy(), hq_dct_cr.numpy()]
            decoded_rgb = jpegT.decode_rgb(hq_dct, subsample=self.default_kwargs["subsample"])

            torchvision.io.write_png(
                torch.from_numpy(decoded_rgb).permute(2,0,1),
                os.path.join(DATAPOINTS_OUTPUT_DIR, name + "-hq_dct.png")
            )

            # Save an image from LQ DCT coeffs
            #
            lq_dct_y = point["lq_dct_y"]
            lq_dct_cb = point["lq_dct_cb"]
            lq_dct_cr = point["lq_dct_cr"]

            lq_dct_y = lq_dct_y.permute(1,2,0)
            lq_dct_cb = lq_dct_cb.permute(1,2,0)
            lq_dct_cr = lq_dct_cr.permute(1,2,0)

            lq_dct_y = lq_dct_y.reshape(lq_dct_y.shape[:-1] + (8,8))
            lq_dct_cb = lq_dct_cb.reshape(lq_dct_cb.shape[:-1] + (8,8))
            lq_dct_cr = lq_dct_cr.reshape(lq_dct_cr.shape[:-1] + (8,8))

            lq_dct = [lq_dct_y.numpy(), lq_dct_cb.numpy(), lq_dct_cr.numpy()]
            decoded_rgb = jpegT.decode_rgb(lq_dct, subsample=self.default_kwargs["subsample"])

            torchvision.io.write_png(
                torch.from_numpy(decoded_rgb).permute(2,0,1),
                os.path.join(DATAPOINTS_OUTPUT_DIR, name + "-lq_dct.png")
            )

    @pytest.mark.parametrize("subsample", (444, 422, 420))
    def test_normalization(self, subsample):
        self.init()

        kwargs = self.default_kwargs.copy()
        kwargs.update(transform_dct=self.transform_dct1,
                      use_lq_ycc=True, use_hq_ycc=True,
                      use_lq_dct=True, use_hq_dct=True, subsample=subsample)

        dataset = DatasetQuantizedJPEG(**kwargs)
        idct = InverseDCT(
            luma_mean=      self.transform_dct1.luma_mean,
            luma_std=       self.transform_dct1.luma_std,
            chroma_mean=    self.transform_dct1.chroma_mean,
            chroma_std=     self.transform_dct1.chroma_std
        )

        for i in range(LEN_DATASET):
            point = dataset[i]
            lq_planes = upsample_chrominance(
                [point["lq_y"][0].numpy(),
                 point["lq_cb"][0].numpy(),
                 point["lq_cr"][0].numpy()], subsample=subsample
            )
            hq_planes = upsample_chrominance(
                [point["hq_y"][0].numpy(),
                 point["hq_cb"][0].numpy(),
                 point["hq_cr"][0].numpy()], subsample=subsample
            )

            lq_dct_y  = point["lq_dct_y"]
            lq_dct_cb = point["lq_dct_cb"]
            lq_dct_cr = point["lq_dct_cr"]

            hq_dct_y  = point["hq_dct_y"]
            hq_dct_cb = point["hq_dct_cb"]
            hq_dct_cr = point["hq_dct_cr"]

            lq_y  = idct(lq_dct_y.unsqueeze(0), chroma=False)[0,0].numpy()
            lq_cb = idct(lq_dct_cb.unsqueeze(0), chroma=True)[0,0].numpy()
            lq_cr = idct(lq_dct_cr.unsqueeze(0), chroma=True)[0,0].numpy()

            hq_y  = idct(hq_dct_y.unsqueeze(0), chroma=False)[0,0].numpy()
            hq_cb = idct(hq_dct_cb.unsqueeze(0), chroma=True)[0,0].numpy()
            hq_cr = idct(hq_dct_cr.unsqueeze(0), chroma=True)[0,0].numpy()

            lq_ycc = upsample_chrominance([lq_y, lq_cb, lq_cr], subsample=subsample)
            hq_ycc = upsample_chrominance([hq_y, hq_cb, hq_cr], subsample=subsample)

            for x, y in zip(lq_planes, lq_ycc):
                assert np.all(np.isclose(x, y, atol=1e-3))

            for x, y in zip(hq_planes, hq_ycc):
                assert np.all(np.isclose(x, y, atol=1e-3))
