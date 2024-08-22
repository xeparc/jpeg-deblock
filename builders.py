import json

import torch
import torch.utils
import torch.utils.data

from dataset import (
    DatasetQuantizedJPEG,
    ToDCTTensor
)
from models import (
    SpectralNet,
    SpectralEncoder,
    InverseDCT,
    ChromeUpsample,
    ChromaCrop,
    ConvertYccToRGB
)


def get_dct_stats(config):
    with open(config.DATA.DCT_STATS_FILEPATH, mode="rt") as f:
        stats = json.load(f)
        return {
            "luma_mean"     : torch.as_tensor(stats["dct_Y_mean"]),
            "luma_std"      : torch.as_tensor(stats["dct_Y_std"]),
            "chroma_mean"   : torch.as_tensor(stats["dct_C_mean"]),
            "chroma_std"    : torch.as_tensor(stats["dct_C_std"])
        }


def build_spectral_net(config):

    blocks = []
    num_blocks = len(config.MODEL.SPECTRAL.DEPTHS)

    in_features = config.MODEL.SPECTRAL.INPUT_DIM
    for i in range(num_blocks):
        encoder = SpectralEncoder(
            in_features=        in_features,
            num_layers=         config.MODEL.SPECTRAL.DEPTHS[i],
            window_size=        config.MODEL.SPECTRAL.WINDOW_SIZES[i],
            d_model=            config.MODEL.SPECTRAL.EMBED_DIMS[i],
            d_qcoeff=           64,
            num_heads=          config.MODEL.SPECTRAL.NUM_HEADS[i],
            d_feedforward=      config.MODEL.SPECTRAL.MLP_DIMS[i],
            dropout=            config.MODEL.SPECTRAL.DROPOUTS[i],
            add_bias_kqv=       config.MODEL.SPECTRAL.QKV_BIAS
        )

        blocks.append(encoder)

        if i < num_blocks - 1:
            embed_upscale = torch.nn.Conv2d(
                in_channels=        config.MODEL.SPECTRAL.EMBED_DIMS[i],
                out_channels=       config.MODEL.SPECTRAL.EMBED_DIMS[i+1],
                kernel_size=        1,
                stride=             1,
                padding=            0,
                dilation=           0,
                bias=               False

            )
            blocks.append(embed_upscale)
            in_features = config.MODEL.SPECTRAL.EMBED_DIMS[i+1]

    # Initialize Inverse DCT Transform
    stats = get_dct_stats(config)
    idct = InverseDCT(
        luma_mean=      stats["luma_mean"],
        luma_std=       stats["luma_std"],
        chroma_mean=    stats["chroma_mean"],
        chroma_std=     stats["chroma_std"]
    )
    blocks.append(idct)

    return SpectralNet(blocks)


def build_chroma_net(config):
    pass


def build_dataloader(config, kind: str):
    assert kind in ("train", "val", "test")

    if kind == "train":
        image_dirs = config.DATA.LOCATIONS.TRAIN
        batch_size = config.TRAIN.BATCH_SIZE // config.DATA.NUM_PATCHES
    elif kind == "val":
        image_dirs = config.DATA.LOCATIONS.VAL
        batch_size = config.VALIDATION.BATCH_SIZE // config.DATA.NUM_PATCHES
    elif kind == "test":
        image_dirs = config.DATA.LOCATIONS.TEST
        batch_size = config.TEST.BATCH_SIZE // config.DATA.NUM_PATCHES


    if config.DATA.NORMALIZE_DCT:
        coeffs = get_dct_stats(config)
        transform_dct = ToDCTTensor(**coeffs)
    else:
        transform_dct = ToDCTTensor()

    dataset = DatasetQuantizedJPEG(
        image_dirs=         image_dirs,
        patch_size=         config.DATA.PATCH_SIZE,
        num_patches=        config.DATA.NUM_PATCHES,
        min_quality=        config.DATA.MIN_QUALITY,
        max_quality=        config.DATA.MAX_QUALITY,
        target_quality=     config.DATA.TARGET_QUALITY,
        subsample=          config.DATA.SUBSAMPLE,
        transform_dct=      transform_dct,
        use_lq_rgb=         config.DATA.USE_LQ_RGB,
        use_lq_ycc=         config.DATA.USE_LQ_YCC,
        use_lq_dct=         config.DATA.USE_LQ_DCT,
        use_hq_rgb=         config.DATA.USE_HQ_RGB,
        use_hq_ycc=         config.DATA.USE_HQ_YCC,
        use_hq_dct=         config.DATA.USE_HQ_DCT,
        use_qtables=        config.DATA.USE_QTABLES,
        seed=               config.SEED
    )

    dataloader = torch.utils.data.DataLoader(
        dataset             = dataset,
        batch_size          = batch_size,
        shuffle             = config.DATA.SHUFFLE,
        num_workers         = config.DATA.NUM_WORKERS,
        collate_fn          = dataset.collate_fn,
        pin_memory          = config.DATA.PIN_MEMORY,
        pin_memory_device   = config.DATA.PIN_MEMORY_DEVICE
    )

    return dataloader
