import json
import functools
import logging
import os
import sys

import torch
import torch.utils
import torch.utils.data

from dataset import (
    DatasetQuantizedJPEG,
)
from models.models import (
    # Transforms
    ConvertRGBToYcc,
    ConvertYccToRGB,
    InverseDCT,
    ToDCTTensor,
    ToQTTensor,
    # Layers
    BlockEncoder,
    BlockDecoder,
    SpectralNet,
    SpectralEncoder,
    SpectralEncoderLayer,
    SpectralTransformer,
    SpectralModel,
    ChromaNet,
    ConvNeXtBlock,
    GradeModel
)
from models import *
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


def build_block_encoder_decoder(config):
    encoder = BlockEncoder(64, out_channels=config.MODEL.SPECTRAL.EMBED_DIM,
                            interaction=config.MODEL.ENCODER_INTERACTION)
    decoder = BlockDecoder(config.MODEL.SPECTRAL.EMBED_DIM, 64,
                           interaction=config.MODEL.DECODER_INTERACTION)
    return encoder, decoder


def build_spectral_model(config):
    num_layers = config.MODEL.SPECTRAL.DEPTH
    layers = []
    for i in range(num_layers):
        encoder_layer = SpectralEncoderLayer(
            window_size=        config.MODEL.SPECTRAL.WINDOW_SIZES[i],
            d_model=            config.MODEL.SPECTRAL.EMBED_DIM,
            num_heads=          config.MODEL.SPECTRAL.NUM_HEADS[i],
            d_feedforward=      config.MODEL.SPECTRAL.MLP_DIMS[i],
            dropout=            config.MODEL.SPECTRAL.DROPOUTS[i],
            add_bias_kqv=       config.MODEL.SPECTRAL.QKV_BIAS
        )
        layers.append(encoder_layer)

    transformer = SpectralTransformer(layers)
    encoder, decoder = build_block_encoder_decoder(config)
    return SpectralModel(encoder, transformer, decoder)


def build_grade_model(config):

    model = GradeModel(
        in_channels=        config.MODEL.GRADE.IN_CHANNELS,
        num_outputs=        config.MODEL.GRADE.NUM_OUTPUTS,
        depths=             config.MODEL.GRADE.DEPTHS,
        dims=               config.MODEL.GRADE.DIMS,
        stem_kernel_size=   config.MODEL.GRADE.STEM_KERNEL_SIZE,
        stem_stride=        config.MODEL.GRADE.STEM_STRIDE
    )
    return model


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
                dilation=           1,
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
    # blocks.append(idct)

    return SpectralNet(blocks, output_transform=idct)


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
        out_transform = ConvertYccToRGB()
    else:
        out_transform = torch.nn.Identity()

    net = ChromaNet(
        stages=             body,
        output_transform=   out_transform,
        in_channels=        config.MODEL.CHROMA.IN_CHANNELS,
        out_channels=       config.MODEL.CHROMA.OUT_CHANNELS,
        kernel_size=        config.MODEL.CHROMA.BODY_KERNEL_SIZE,
        skip=               config.MODEL.CHROMA.SKIP
    )

    return net


def build_model(config):

    if config.MODEL.CLASS == "MobileNetIR":
        model = MobileNetIR(
            in_channels=        config.MODEL.MOBILENETIR.IN_CHANNELS,
            out_channels=       config.MODEL.MOBILENETIR.OUT_CHANNELS,
        )
    elif config.MODEL.CLASS == "RRDBNet":
        model = RRDBNet(
            luma_blocks=        config.MODEL.RRDBNET.LUMA_BLOCKS,
            chroma_blocks=      config.MODEL.RRDBNET.CHROMA_BLOCKS
        )
    elif config.MODEL.CLASS == "PrismLumaS4":
        dct_stats = get_dct_stats(config)
        idct = InverseDCT(**dct_stats)
        kwargs = dict(config.MODEL.KWARGS)
        model = PrismLumaS4(idct, **kwargs)
    else:
        raise NotImplementedError

    return model


def build_dataloader(config, kind: str):
    assert kind in ("train", "val", "test")

    if kind == "train":
        image_dirs =    config.DATA.LOCATIONS.TRAIN
        num_patches =   config.DATA.NUM_PATCHES
        batch_size =    config.TRAIN.BATCH_SIZE // num_patches
        region_size =   config.DATA.REGION_SIZE
        patch_size =    config.DATA.PATCH_SIZE
    elif kind == "val":
        image_dirs =    config.DATA.LOCATIONS.VAL
        num_patches =   1
        batch_size =    config.VALIDATION.BATCH_SIZE
        region_size =   config.DATA.REGION_SIZE
        patch_size =    config.DATA.PATCH_SIZE
    elif kind == "test":
        image_dirs =    config.DATA.LOCATIONS.TEST
        num_patches =   1
        batch_size =    config.TEST.BATCH_SIZE
        region_size =   400
        patch_size =    400

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

    device = torch.device(config.TRAIN.DEVICE)

    dataloader = torch.utils.data.DataLoader(
        dataset             = dataset,
        batch_size          = batch_size,
        shuffle             = config.DATA.SHUFFLE,
        num_workers         = config.DATA.NUM_WORKERS,
        collate_fn          = functools.partial(dataset.collate_fn, device=device),
        pin_memory          = config.DATA.PIN_MEMORY,
        pin_memory_device   = config.DATA.PIN_MEMORY_DEVICE if config.DATA.PIN_MEMORY else ''
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
        luma_weight=    config.LOSS.LUMA_WEIGHT,
        chroma_weight=  config.LOSS.CHROMA_WEIGHT,
        alpha=          config.LOSS.ALPHA,
        beta=           config.LOSS.BETA
    )


def build_optimizer(config, *params):

    name = config.TRAIN.OPTIMIZER.NAME
    kwargs = {k:v for k,v in config.TRAIN.OPTIMIZER.KWARGS}

    if name == "adamw":
        optim = torch.optim.AdamW(
            params=[{"params": p} for p in params],
            lr=config.TRAIN.BASE_LR,
            **kwargs
        )
    elif name == "sgd":
        optim = torch.optim.SGD(
            params=[{"params": p} for p in params],
            lr=config.TRAIN.BASE_LR,
            **kwargs
        )
    elif name == "rmsprop":
        optim = torch.optim.RMSprop(
            params=[{"params": p} for p in params],
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

    if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX:
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
    logger.warning("\n\n\t\t=== > STARTING TRAINING === >\n")

    return logger
