import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Datatest config
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = "DIV2K"
_C.DATA.LOCATIONS_TRAIN = ["data/DIV2K/DIV2K_train_HR/"]
_C.DATA.LOCATIONS_VAL = ["data/DIV2K/DIV2K_valid_HR/"]
_C.DATA.LOCATION_TEST = []
_C.DATA.PATCH_SIZE = 64
_C.DATA.NUM_PATCHES = 64
_C.DATA.SUBSAMPLE = "420"
_C.DATA.MIN_QUALITY = 10
_C.DATA.MAX_QUALITY = 85
_C.DATA.TARGET_QUALITY = 100
_C.DATA.CACHED = True
_C.DATA.USE_LQ_RGB = False
_C.DATA.USE_LQ_YCC = False
_C.DATA.USE_LQ_DCT = True
_C.DATA.USE_HQ_RGB = False
_C.DATA.USE_HQ_YCC = True
_C.DATA.USE_HQ_DCT = True
_C.DATA.USE_QTABLES = True
_C.DATA.NORMALIZE_RGB = False
_C.DATA.NORMALIZE_YCC = False
_C.DATA_NORMALIZE_DCT = True
_C.DATA_DCT_STATS_FILEPATH = "data/DIV2K+Flickr2K-dct-stats.json"
_C.DATA.PIN_MEMORY = True
_C.DATA.PIN_MEMORY_DEVICE = "mps"


# -----------------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = ""


# # Swin Transformer parameters
# _C.MODEL.SWIN = CN()
# _C.MODEL.SWIN.PATCH_SIZE = 4
# _C.MODEL.SWIN.IN_CHANS = 3
# _C.MODEL.SWIN.EMBED_DIM = 96
# _C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
# _C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
# _C.MODEL.SWIN.WINDOW_SIZE = 7
# _C.MODEL.SWIN.MLP_RATIO = 4.
# _C.MODEL.SWIN.QKV_BIAS = True
# _C.MODEL.SWIN.QK_SCALE = None
# _C.MODEL.SWIN.APE = False
# _C.MODEL.SWIN.PATCH_NORM = True


# Localized Frequency Domain Transformer
# Number of
_C.MODEL.LFDT.INPUT_DIM = 64
_C.MODEL.LFDT.DEPTHS = [1, 4, 2]
_C.MODEL.LFDT.EMBED_DIMS = [64, 128, 64]
_C.MODEL.LFDT.WINDOW_SIZES = [7, 7, 7]
_C.MODEL.LFDT.NUM_HEADS = [4, 4, 4]
_C.MODEL.LFDT.MLP_DIMS = [512, 1024, 1024]
_C.MODEL.LFDT.QKV_BIAS = True
_C.MODEL.LFDT.DROPOUTS = [0.1, 0.1, 0.1]

# Frequency Enhance Net -> follows after LFDT

# Enhanced Chroma Upscale Net
_C.MODEL.ECUN.DEPTHS = [1, 1, 1]
_C.MODEL.ECUN.CHANNELS = [32, 64, 3]


# -----------------------------------------------------------------------------
# Train config
# -----------------------------------------------------------------------------
_C.TRAIN.BATCH_SIZE = 32
