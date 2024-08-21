import json

import torch

from dataset import DatasetQuantizedJPEG
from models import (
    FDNet,
    LFDTEncoder,
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


def build_FDNet(config):

    blocks = []
    num_blocks = config.MODEL.LFDT.DEPTHS

    in_features = config.MODEL.LFDT.INPUT_DIM
    for i in range(num_blocks):
        encoder = LFDTEncoder(
            in_features=        in_features,
            num_layers=         config.MODEL.LFDT.DEPTHS[i],
            window_size=        config.MODEL.LFDT.WINDOW_SIZES[i],
            d_model=            config.MODEL.LFDT.EMBED_DIMS[i],
            d_qcoeff=           64,
            num_heads=          config.MODEL.LFDT.NUM_HEADS[i],
            d_feedforward=      config.MODEL.LFDT.MLP_DIMS[i],
            dropout=            config.MODEL.LFDT.DROPOUTS[i],
            add_bias_kqv=       config.MODEL.LFDT.QKV_BIAS
        )

        blocks.append(encoder)

        if i < num_blocks:
            embed_upscale = torch.nn.Conv2d(
                in_channels=        config.MODEL.LFDT.EMBED_DIMS[i],
                out_channels=       config.MODEL.LFDT.EMBED_DIMS[i+1],
                kernel_size=        1,
                stride=             1,
                padding=            0,
                dilation=           0,
                bias=               False

            )
            blocks.append(embed_upscale)
            in_features = config.MODEL.LFDT.EMBED_DIMS[i+1]

    # Initialize Inverse DCT Transform
    stats = get_dct_stats(config)
    idct = InverseDCT(
        luma_mean=      stats["luma_mean"],
        luma_std=       stats["luma_std"],
        chroma_mean=    stats["chroma_mean"],
        chroma_std=     stats["chroma_std"]
    )
    blocks.append(idct)

    return FDNet(blocks)

