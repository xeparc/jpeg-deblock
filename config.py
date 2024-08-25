import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

_C.SEED = 7
_C.TAG = "default"

# -----------------------------------------------------------------------------
# Datatest config
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = "DIV2K"
_C.DATA.LOCATIONS = CN()
_C.DATA.LOCATIONS.TRAIN = ["data/DIV2K/DIV2K_train_HR/", "data/Flickr2K/"]
_C.DATA.LOCATIONS.VAL = ["data/DIV2K/DIV2K_valid_HR/"]
_C.DATA.LOCATIONS.TEST = ["data/10/", "data/20/", "data/30/", "data/40/"]
_C.DATA.PATCH_SIZE = 64
_C.DATA.NUM_PATCHES = 16
_C.DATA.SUBSAMPLE = 420
_C.DATA.MIN_QUALITY = 10
_C.DATA.MAX_QUALITY = 85
_C.DATA.TARGET_QUALITY = 100
_C.DATA.CACHED = False
_C.DATA.USE_LQ_RGB = False
_C.DATA.USE_LQ_YCC = False
_C.DATA.USE_LQ_DCT = True
# Include target image's RGB channels in datapoints ?
_C.DATA.USE_HQ_RGB = True
# Include target image's YCbCr planes in datapoints ?
_C.DATA.USE_HQ_YCC = True
# Include target image's DCT in datapoints ?
_C.DATA.USE_HQ_DCT = False
# Include JPEG quantization tables in datapoints ?
_C.DATA.USE_QTABLES = True
_C.DATA.NORMALIZE_RGB = False
_C.DATA.NORMALIZE_YCC = False
_C.DATA.NORMALIZE_DCT = True
_C.DATA.INVERT_QT = False
_C.DATA.DCT_STATS_FILEPATH = "data/DIV2K+Flickr2K-dct-stats.json"
_C.DATA.PIN_MEMORY = False
_C.DATA.PIN_MEMORY_DEVICE = "mps"
_C.DATA.SHUFFLE = True
_C.DATA.NUM_WORKERS = 0


# -----------------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.MODEL.RGB_OUTPUT = True
_C.MODEL.RESUME = ''

# Localized Frequency Domain Transformer (LFDT) parameters
# We'll call it Spectral Transformer :)
_C.MODEL.SPECTRAL = CN()
# Number of
_C.MODEL.SPECTRAL.INPUT_DIM = 64
_C.MODEL.SPECTRAL.DEPTHS = [1, 4, 2]
_C.MODEL.SPECTRAL.EMBED_DIMS = [64, 128, 64]
_C.MODEL.SPECTRAL.WINDOW_SIZES = [7, 7, 7]
_C.MODEL.SPECTRAL.NUM_HEADS = [4, 4, 4]
_C.MODEL.SPECTRAL.MLP_DIMS = [512, 1024, 1024]
_C.MODEL.SPECTRAL.QKV_BIAS = True
_C.MODEL.SPECTRAL.DROPOUTS = [0.1, 0.1, 0.1]

# Frequency Enhance Net -> follows after LFDT

# Chrominance Upscale Net
# We'll call it Chroma net
_C.MODEL.CHROMA = CN()
_C.MODEL.CHROMA.SKIP = False
_C.MODEL.CHROMA.DEPTHS = [1, 2, 1]
_C.MODEL.CHROMA.CHANNELS = [32, 64, 32]
_C.MODEL.CHROMA.CHANNEL_MULTIPLIER = 2
_C.MODEL.CHROMA.IN_CHANNELS = 3
_C.MODEL.CHROMA.OUT_CHANNELS = 3
_C.MODEL.CHROMA.STEM_KERNEL_SIZE = 3
_C.MODEL.CHROMA.BODY_KERNEL_SIZE = 3
_C.MODEL.CHROMA.LEAF_KERNEL_SIZE = 3


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.DEVICE = "mps"
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.START_ITERATION = 0
_C.TRAIN.NUM_ITERATIONS = 50_000
_C.TRAIN.WARMUP_ITERATIONS = 1000
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 1e-6
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.CHECKPOINT_EVERY = 2000
_C.TRAIN.CHECKPOINT_DIR = "checkpoints/"
# _C.TRAIN.ACCUMULATION_STEPS = 1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
_C.TRAIN.OPTIMIZER.KWARGS = [ ("betas", (0.9, 0.999)), ("weight_decay", 1e-3) ]

# Learning rate scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
_C.TRAIN.LR_SCHEDULER.KWARGS = [("T_0", 10_000), ("eta_min", 1e-6), ("T_mult", 3)]
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True


# -----------------------------------------------------------------------------
# Validation config
# -----------------------------------------------------------------------------
_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE = 64

# -----------------------------------------------------------------------------
# Test config
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64


# -----------------------------------------------------------------------------
# Loss config
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Name of the loss function / criterion used from PyTorch library
_C.LOSS.CRITERION = "huber"
# Keyword arguments passed to torch.nn loss criterion
_C.LOSS.CRITERION_KWARGS = [("delta", 1.0)]
# Multiplier of Luminance (Y) plane loss in total Spectral Loss
_C.LOSS.LUMA_WEIGHT = 0.5
# Multiplier of Chrominance (Cb, Cr) planes loss in total Spectral Loss
_C.LOSS.CHROMA_WEIGHT = 0.25
# Multiplier of Spectral Loss
_C.LOSS.ALPHA = 100.0
# Multiplier of Chroma Loss
_C.LOSS.BETA = 1.0
# -----------------------------------------------------------------------------
# Logging setting
# -----------------------------------------------------------------------------
_C.LOGGING = CN()
_C.LOGGING.LOG_EVERY = 100
_C.LOGGING.DIR = "logs/"



def default_config():
    return _C.clone()


def get_config(filename):
    cfg = default_config()
    cfg.merge_from_file(filename)
    return cfg


def update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()
