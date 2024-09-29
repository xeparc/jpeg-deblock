import json
import logging
import os
import sys

import torch
import torch.utils
import torch.utils.data

from dataset import (
    DatasetQuantizedJPEG,
)
from models.blocks import (
    InverseDCT,
    ToDCTTensor,
    ToQTTensor,
)
from models import *


def get_dct_stats(config):
    with open(config.DATA.DCT_STATS_FILEPATH, mode="rt") as f:
        stats = json.load(f)
        return {
            "luma_mean"     : torch.as_tensor(stats["dct_Y_mean"]),
            "luma_std"      : torch.as_tensor(stats["dct_Y_std"]),
            "chroma_mean"   : torch.as_tensor(stats["dct_C_mean"]),
            "chroma_std"    : torch.as_tensor(stats["dct_C_std"])
        }


def build_idct(config):
    """Builds Inverse DCT transform Module."""
    if config.DATA.NORMALIZE_DCT:
        with open(config.DATA.DCT_STATS_FILEPATH, mode="rt") as f:
            stats = json.load(f)
            return InverseDCT(
                luma_mean     = torch.as_tensor(stats["dct_Y_mean"]),
                luma_std      = torch.as_tensor(stats["dct_Y_std"]),
                chroma_mean   = torch.as_tensor(stats["dct_C_mean"]),
                chroma_std    = torch.as_tensor(stats["dct_C_std"])
            )
    else:
        return InverseDCT()


def build_model(config):

    if config.MODEL.CLASS == "MobileNetIR":
        model = MobileNetIR(**dict(config.MODEL.KWARGS))

    elif config.MODEL.CLASS == "RRDBNet":
        model = RRDBNet(**dict(config.MODEL.KWARGS))

    elif config.MODEL.CLASS == "Prism":
        dct_stats = get_dct_stats(config)
        idct = InverseDCT(**dct_stats)
        kwargs = dict(config.MODEL.KWARGS)
        model = Prism(idct, **kwargs)

    elif config.MODEL.CLASS == "FlareLuma":
        dct_stats = get_dct_stats(config)
        idct = InverseDCT(**dct_stats)
        kwargs = dict(config.MODEL.FLARE.LUMA.KWARGS)
        model = FlareLuma(idct, **kwargs)

    elif config.MODEL.CLASS == "Flare":
        dct_stats = get_dct_stats(config)
        idct = InverseDCT(**dct_stats)
        # Build Luma submodel
        luma_kwargs = dict(config.MODEL.FLARE.LUMA.KWARGS)
        luma = FlareLuma(idct, **luma_kwargs)
        # Build Chroma submodel
        chroma_kwargs = dict(config.MODEL.FLARE.CHROMA.KWARGS)
        chroma = FlareChroma(idct, **chroma_kwargs)
        # Build Flare model
        flare_kwargs = dict(config.MODEL.FLARE.KWARGS)
        model = Flare(luma, chroma, **flare_kwargs)

    elif config.MODEL.CLASS == "MobileNetQA":
        if config.MODEL.INPUTS[0] in ("lq_y", "lq_cb", "lq_cr"):
            model = MobileNetQA(in_channels=1)
        elif config.MODEL.INPUTS[0] in ("lq_ycc", "lq_rgb"):
            model = MobileNetQA(in_channels=3)
        else:
            raise NotImplementedError("Unsupported inputs: " + str(config.MODEL.INPUTS))

    elif config.MODEL.CLASS == "Q1Net":
        if config.MODEL.INPUTS[0] in ("lq_y", "lq_cb", "lq_cr"):
            in_channels = 1
        elif config.MODEL.INPUTS[0] in ("lq_ycc", "lq_rgb"):
            in_channels = 3
        else:
            raise NotImplementedError("Unsupported inputs: " + str(config.MODEL.INPUTS))
        model = Q1Net(in_channels)
    else:
        raise NotImplementedError

    return model


def build_dataloader(config, kind: str, quality: int = 0):
    assert kind in ("train", "val", "test")

    if kind == "train":
        image_dirs =    config.DATA.LOCATIONS.TRAIN
        num_patches =   config.DATA.NUM_PATCHES
        batch_size =    config.TRAIN.BATCH_SIZE // num_patches
        region_size =   config.DATA.REGION_SIZE
        patch_size =    config.DATA.PATCH_SIZE
        cached =        config.DATA.CACHED
    elif kind == "val":
        image_dirs =    config.DATA.LOCATIONS.VAL
        num_patches =   1
        batch_size =    config.VALIDATION.BATCH_SIZE
        region_size =   config.DATA.PATCH_SIZE
        patch_size =    config.DATA.PATCH_SIZE
        cached =        False
    elif kind == "test":
        image_dirs =    config.DATA.LOCATIONS.TEST
        num_patches =   1
        batch_size =    config.TEST.BATCH_SIZE
        region_size =   config.TEST.REGION_SIZE
        patch_size =    config.TEST.REGION_SIZE
        cached =        False

    # Optionally, override min/max quaility in config.DATA.
    # This is used to evaluate models on test samples.
    if quality > 0:
        min_quality = quality
        max_quality = quality + 1
    else:
        min_quality = config.DATA.MIN_QUALITY
        max_quality = config.DATA.MAX_QUALITY

    if config.DATA.NORMALIZE_DCT:
        coeffs = get_dct_stats(config)
        transform_dct = ToDCTTensor(**coeffs)
    else:
        transform_dct = ToDCTTensor()

    transform_qt = ToQTTensor(config.DATA.INVERT_QT)

    dataset = DatasetQuantizedJPEG(
        image_dirs=         image_dirs,
        region_size=        region_size,
        patch_size=         patch_size,
        num_patches=        num_patches,
        min_quality=        min_quality,
        max_quality=        max_quality,
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
        seed=               config.SEED,
        cached=             cached,
        cache_memory=       config.DATA.CACHE_MEMORY
    )

    device = torch.device(config.TRAIN.DEVICE)

    dataloader = torch.utils.data.DataLoader(
        dataset             = dataset,
        batch_size          = batch_size,
        shuffle             = config.DATA.SHUFFLE,
        num_workers         = config.DATA.NUM_WORKERS,
        collate_fn          = dataset.collate_fn,
        pin_memory          = config.DATA.PIN_MEMORY,
    )

    return dataloader


def build_criterion(config):
    kwargs = {k:v for k,v in config.LOSS.KWARGS}
    if config.LOSS.CRITERION == "HuberLoss":
        criterion = torch.nn.HuberLoss(**kwargs)
    elif config.LOSS.CRITERION == "MSELoss":
        criterion = torch.nn.MSELoss(**kwargs)
    elif config.LOSS.CRITERION == "MixedQ1MSELoss":
        criterion = MixedQ1MSELoss(**kwargs)
    else:
        raise NotImplementedError

    return criterion


def build_optimizer(config, *params_list):

    name = config.TRAIN.OPTIMIZER.NAME
    kwargs = {k:v for k,v in config.TRAIN.OPTIMIZER.KWARGS}

    # Collect parameters with `require_grad=True`
    trainable_params = []
    for param in params_list:
        trainable_params.extend(p for p in param if p.requires_grad)

    if name == "adamw":
        optim = torch.optim.AdamW(
            params=trainable_params,
            lr=config.TRAIN.BASE_LR,
            **kwargs
        )
    elif name == "sgd":
        optim = torch.optim.SGD(
            params=trainable_params,
            lr=config.TRAIN.BASE_LR,
            **kwargs
        )
    elif name == "rmsprop":
        optim = torch.optim.RMSprop(
            params=trainable_params,
            lr=config.TRAIN.BASE_LR,
            **kwargs
        )
    else:
        raise NotImplementedError

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

    if config.TRAIN.WARMUP_ITERATIONS > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.TRAIN.WARMUP_LR / (config.TRAIN.BASE_LR),
            end_factor=1.0,
            total_iters=config.TRAIN.WARMUP_ITERATIONS
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.TRAIN.WARMUP_ITERATIONS]
        )
    else:
        return main_scheduler


def build_logger(config, name="train"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create formatter
    info_fmt = "[%(asctime)s]: %(levelname)s %(message)s"
    debug_fmt = "[%(asctime)s] (%(filename)s): %(levelname)s %(message)s"
    warning_fmt = "[%(asctime)s]: %(message)s"
    datefmt = '%Y-%m-%d %H:%M:%S'

    # Create file handlers
    savepath_info = os.path.join(config.LOGGING.DIR, config.TAG, "info.txt")
    savepath_debug = os.path.join(config.LOGGING.DIR, config.TAG, "debug.txt")

    #   - Create info handler
    if not os.path.exists(savepath_info):
        os.makedirs(os.path.dirname(savepath_info), exist_ok=True)
        open(savepath_info, mode='w').close()
    info_handler = logging.FileHandler(savepath_info, mode='a')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter(info_fmt, datefmt))

    #   - Create debug handler
    if not os.path.exists(savepath_debug):
        os.makedirs(os.path.dirname(savepath_debug), exist_ok=True)
        open(savepath_debug, mode='w').close()
    debug_handler = logging.FileHandler(savepath_debug, mode='a')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(debug_fmt, datefmt))

    #   - Create console handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter(warning_fmt, datefmt))

    logger.addHandler(info_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    pid = os.getpid()
    logger.warning("\n\n\t\t=== > STARTING TRAINING === >\n")
    logger.warning("\t\tProcess PID: ", pid, "\n\n")
    return logger
