import json
import logging
import os

import torch
import torch.utils
import torch.utils.data

from dataset import (
    DatasetQuantizedJPEG,
    ToDCTTensor,
    ToQTTensor
)
from models import (
    SpectralNet,
    SpectralEncoder,
    ChromaNet,
    ConvNeXtBlock,
    InverseDCT,
    ConvertYccToRGB
)
from utils import CombinedLoss


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
    
    depths =    config.MODEL.CHROMA.DEPTHS
    channels =  config.MODEL.CHROMA.CHANNELS

    num_blocks = len(depths)
    body = []
    for i in range(num_blocks):
        for _ in range(depths[i]):
            block = ConvNeXtBlock(
                in_channels=    channels[i],
                mid_channels=   config.MODEL.CHROMA.CHANNEL_MULTIPLIER * channels[i],
                kernel_size=    config.MODEL.CHROMA.BODY_KERNEL_SIZE
            )
            body.append(block)
    
    if config.MODEL.RGB_OUTPUT:
        out_transform = ConvertYccToRGB
    else:
        out_transform = torch.nn.Identity

    net = ChromaNet(
        stages=             body,
        output_transform=   out_transform,
        in_channels=        config.MODEL.CHROMA.IN_CHANNELS,
        out_channels=       config.MODEL.CHROMA.OUT_CHANNELS,
        kernel_size=        config.MODEL.CHROMA.STEM_KERNEL_SIZE,
    )

    return net


def build_dataloader(config, kind: str):
    assert kind in ("train", "val", "test")

    if kind == "train":
        image_dirs = config.DATA.LOCATIONS.TRAIN
        num_patches = config.DATA.NUM_PATCHES
        batch_size = config.TRAIN.BATCH_SIZE // num_patches
    elif kind == "val":
        image_dirs = config.DATA.LOCATIONS.VAL
        num_patches = 1
        batch_size = config.VALIDATION.BATCH_SIZE
    elif kind == "test":
        image_dirs = config.DATA.LOCATIONS.TEST
        num_patches = 1
        batch_size = config.TEST.BATCH_SIZE 

    if config.DATA.NORMALIZE_DCT:
        coeffs = get_dct_stats(config)
        transform_dct = ToDCTTensor(**coeffs)
    else:
        transform_dct = ToDCTTensor()

    transform_qt = ToQTTensor(config.DATA.INVERT_QT)

    dataset = DatasetQuantizedJPEG(
        image_dirs=         image_dirs,
        patch_size=         config.DATA.PATCH_SIZE,
        num_patches=        num_patches,
        min_quality=        config.DATA.MIN_QUALITY,
        max_quality=        config.DATA.MAX_QUALITY,
        target_quality=     config.DATA.TARGET_QUALITY,
        subsample=          config.DATA.SUBSAMPLE,
        transform_dct=      transform_dct,
        transform_qt=       transform_qt,
        use_lq_rgb=         config.DATA.USE_LQ_RGB,
        use_lq_ycc=         config.DATA.USE_LQ_YCC,
        use_lq_dct=         config.DATA.USE_LQ_DCT,
        use_hq_rgb=         config.DATA.USE_HQ_RGB,
        use_hq_ycc=         config.DATA.USE_HQ_YCC,
        use_hq_dct=         config.DATA.USE_HQ_DCT,
        use_qt=             config.DATA.USE_QTABLES,
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


def build_criterion(config):
    kwargs = {k:v for k,v in config.LOSS.CRITERION_KWARGS}
    if config.LOSS.CRITERION == "huber":
        criterion = torch.nn.HuberLoss(**kwargs)
    elif config.LOSS.CRITERION == "mse":
        criterion = torch.nn.MSELoss(**kwargs)
    else:
        raise ValueError

    return CombinedLoss(
        criterion=      criterion,
        gamma=          config.LOSS.GAMMA,
        luma_weight=    config.LOSS.LUMA_WEIGHT,
        chroma_weight=  config.LOSS.CHROMA_WEIGHT
    )


def build_optimizer(config, spectral_params, chroma_params):

    name = config.TRAIN.OPTIMIZER.NAME
    kwargs = {k:v for k,v in config.TRAIN.OPTIMIZER.KWARGS}

    if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX:
        lr = config.TRAIN.WARMUP_LR
    else:
        lr = config.TRAIN.BASE_LR

    if name == "adamw":
        optim = torch.optim.AdamW(
            params=[{"params": spectral_params, "params": chroma_params}],
            lr=lr,
            **kwargs
        )
    return optim


def build_lr_scheduler(config, optimizer):

    kwargs = {k:v for k,v in config.TRAIN.LR_SCHEDULER.KWARGS}
    if config.TRAIN.LR_SCHEDULER.NAME == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            **kwargs
        )
    elif config.TRAIN.LR_SCHEDULER_NAME == "multistep":
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            **kwargs
        )
    elif config.TRAIN_LR_SCHEDULER_NAME == "plateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **kwargs
        )
    else:
        raise ValueError

    if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX:
        warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=config.TRAIN.WARMUP_ITERATIONS
        )
        return torch.optim.lr_scheduler.ChainedScheduler([
            warmup_scheduler, main_scheduler
        ])
    else:
        return main_scheduler


def build_logger(config, name="train"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

    # create file handlers
    savepath = os.path.join(config.LOGGING.DIR, "log.txt")
    file_handler = logging.FileHandler(savepath, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
