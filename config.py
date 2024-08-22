import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

_C.SEED = 7

# -----------------------------------------------------------------------------
# Datatest config
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = "DIV2K"
_C.DATA.LOCATIONS = CN()
_C.DATA.LOCATIONS.TRAIN = ["data/DIV2K/DIV2K_train_HR/"]
_C.DATA.LOCATIONS.VAL = ["data/DIV2K/DIV2K_valid_HR/"]
_C.DATA.LOCATIONS.TEST = ["data/10/", "data/20/", "data/30/", "data/40/"]
_C.DATA.PATCH_SIZE = 64
_C.DATA.NUM_PATCHES = 16
_C.DATA.SUBSAMPLE = "420"
_C.DATA.MIN_QUALITY = 10
_C.DATA.MAX_QUALITY = 85
_C.DATA.TARGET_QUALITY = 100
_C.DATA.CACHED = True
_C.DATA.USE_LQ_RGB = False
_C.DATA.USE_LQ_YCC = False
_C.DATA.USE_LQ_DCT = True
# Include target image's RGB channels in datapoints ?
_C.DATA.USE_HQ_RGB = False
# Include target image's YCbCr planes in datapoints ?
_C.DATA.USE_HQ_YCC = True
# Include target image's DCT in datapoints ?
_C.DATA.USE_HQ_DCT = True
# Include JPEG quantization tables in datapoints ?
_C.DATA.USE_QTABLES = True
_C.DATA.NORMALIZE_RGB = False
_C.DATA.NORMALIZE_YCC = False
_C.DATA.NORMALIZE_DCT = True
_C.DATA.DCT_STATS_FILEPATH = "data/DIV2K+Flickr2K-dct-stats.json"
_C.DATA.PIN_MEMORY = True
_C.DATA.PIN_MEMORY_DEVICE = "mps"
_C.DATA.SHUFFLE = True
_C.DATA.NUM_WORKERS = 1


# -----------------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = ""

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
_C.MODEL.CHROMA.DEPTHS = [1, 1, 1]
_C.MODEL.CHROMA.CHANNELS = [32, 64, 3]


# -----------------------------------------------------------------------------
# Train config
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32


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


def default_config():
    return _C.clone()