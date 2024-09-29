import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# Seed
_C.SEED = 7
# Train run tag. Used to initialize log and checkpoint directories
_C.TAG = ""
# Train run comment. Free text
_C.COMMENT = ""

# -----------------------------------------------------------------------------
# Datatest config
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Dataset name
_C.DATA.NAME = "DIV2K"
_C.DATA.LOCATIONS = CN()
# List of dirs with train images
_C.DATA.LOCATIONS.TRAIN = ["data/DIV2K/DIV2K_train_HR/", "data/Flickr2K/"]
# List of dirs with validation images
_C.DATA.LOCATIONS.VAL = ["data/DIV2K/DIV2K_valid_HR/"]
# List of dirs with test images
_C.DATA.LOCATIONS.TEST = ["data/10/", "data/20/", "data/30/", "data/40/"]
# Size of the central square region from which patches are sampled.
# If negative, the CenterCrop transform is ignored.
_C.DATA.REGION_SIZE = 800
# Size of extracted patch
_C.DATA.PATCH_SIZE = 64
# Number of extracted patches from single image
_C.DATA.NUM_PATCHES = 16
# Chrominance subsampling mode
_C.DATA.SUBSAMPLE = 420
# Minimum sampled JPEG quality
_C.DATA.MIN_QUALITY = 10
# Maximum sampled JPEG quality
_C.DATA.MAX_QUALITY = 85
# Target JPEG quality
_C.DATA.TARGET_QUALITY = 100
# If True, images are loaded in memory for faster access
_C.DATA.CACHED = False
# Maximum amount of RAM dedicated to cache (in GB)
_C.DATA.CACHE_MEMORY = 16
# Include RGB channels from LQ image in datapoint ?
_C.DATA.USE_LQ_RGB = False
# Include YCbCr planes from LQ image in datapoint ?
_C.DATA.USE_LQ_YCC = False
# Include DCT coefficients from LQ image in datapoint ?
_C.DATA.USE_LQ_DCT = True
# Include RGB channels from HQ image in datapoint ?
_C.DATA.USE_HQ_RGB = True
# Include YCbCr planes from HQ image in datapoint ?
_C.DATA.USE_HQ_YCC = True
# Include DCT coefficients from HQ image in datapoint ?
_C.DATA.USE_HQ_DCT = False
# Include JPEG quantization tables in datapoint ?
_C.DATA.USE_QTABLES = True
# Normalize DCT coefficients ?
_C.DATA.NORMALIZE_DCT = True
# If True, map [1, 255] quantization table range to [1, 0].
# If False, map [1, 255] quantization table range to [0, 1].
_C.DATA.INVERT_QT = False
# Path to JSON file with DCT coefficients mean and std. Used for normalization.
_C.DATA.DCT_STATS_FILEPATH = "data/DIV2K+Flickr2K-dct-stats.json"
# Pin memory on host device ?
_C.DATA.PIN_MEMORY = False
# Shuffle data in dataloader
_C.DATA.SHUFFLE = True
# Number of workers in Torch DataLoader class
_C.DATA.NUM_WORKERS = 0


# -----------------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name. Free text
_C.MODEL.NAME = ""
# Class name used to instantiate model. Should match already deined class.
_C.MODEL.CLASS = ""
# Keys used to extact model inputs from batch. Passed to `forward()`
_C.MODEL.INPUTS = ["lq_rgb"]
# Keys used to extact targets from batch. Passed to loss criterion
_C.MODEL.TARGETS = ["hq_rgb"]
# Keyowrd arguments passed to MODEL.CLASS constructor
_C.MODEL.KWARGS = []

#---    Flare Luma / Chroma Models
_C.MODEL.FLARE = CN()
_C.MODEL.FLARE.LUMA = CN()
_C.MODEL.FLARE.CHROMA = CN()
# Keyword arguments passed to `FlareNet.__init__()`
_C.MODEL.FLARE.KWARGS = []
# Keyword arguments passed to `FlareLuma.__init__()`
_C.MODEL.FLARE.LUMA.KWARGS = []
# Keyword arguments passed to `FlareChroma.__init__()`
_C.MODEL.FLARE.CHROMA.KWARGS = []


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Device used for training
_C.TRAIN.DEVICE = "mps"
# Path to checkpoint file, from which to resume training
_C.TRAIN.RESUME = ""
# Train batch size
_C.TRAIN.BATCH_SIZE = 32
# Update parameters once in every `ACCUMULATE_GRADS` iterations.
# This is used to simulate larger batch sizes.
_C.TRAIN.ACCUMULATE_GRADS = 1
# Start iteration. If training is resumed from checkpoint, this number
# should be the number of iterations from last train session.
_C.TRAIN.START_ITERATION = 0
# Number of train iterations for current run
_C.TRAIN.NUM_ITERATIONS = 100_000
# Number of warmup iterations
_C.TRAIN.WARMUP_ITERATIONS = 10_000
# Learning rate after warmup stage
_C.TRAIN.BASE_LR = 5e-4
# Initial learning rate.
_C.TRAIN.WARMUP_LR = 1e-6
# Maximum gradient norm. Gradients with norm > this one, are clipped
_C.TRAIN.CLIP_GRAD = 5.0
# How the norm the gradients computed?
#   * If "total", the norm is computed over all gradients together, as if they
# were concatenated into a single vector.
#   * If "param", the norm is computed individually for each parameter.
_C.TRAIN.CLIP_GRAD_METHOD = "param"
# Number of iterations after which checkpoint files are saved
_C.TRAIN.CHECKPOINT_EVERY = 100
# Parent directory where checkpoint files are saved.
# The directory of checkpoint files is: CHECKPOINT_DIR/TAG/<iter>
_C.TRAIN.CHECKPOINT_DIR = "checkpoints/"


# -----------------------------------------------------------------------------
# Optimizer & LR scheduler settings
# -----------------------------------------------------------------------------
_C.TRAIN.OPTIMIZER = CN()
# Optimizer name
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Keyword arguments passed to optimizer `__init__()`
_C.TRAIN.OPTIMIZER.KWARGS = [ ("betas", (0.9, 0.999)), ("weight_decay", 1e-6) ]

_C.TRAIN.LR_SCHEDULER = CN()
# Learning rate scheduler name
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Keyword arguments passed to scheduler's `__init__()`
_C.TRAIN.LR_SCHEDULER.KWARGS = [("T_0", 99_000), ("eta_min", 1e-6), ("T_mult", 3)]

# -----------------------------------------------------------------------------
# Loss config
# -----------------------------------------------------------------------------
_C.TRAIN.LOSS = CN()
# Class name used to instantiate loss criterion
_C.TRAIN.LOSS.CRITERION = "MSELoss"
# Keyword arguments passed to criterion constructor
_C.TRAIN.LOSS.KWARGS = []


# -----------------------------------------------------------------------------
# Validation config
# -----------------------------------------------------------------------------
_C.VALIDATION = CN()
# Batch size during validation
_C.VALIDATION.BATCH_SIZE = 64
# Run validation procedure after this many iterations
_C.VALIDATION.EVERY = 500
# Run validation procedure for JPEG images compressed with these quality factors
_C.VALIDATION.QUALITIES = [10, 20, 40, 60, 80]


# -----------------------------------------------------------------------------
# Test config
# -----------------------------------------------------------------------------
_C.TEST = CN()
# If `True` run test procedure during training for images in DATA.LOCATIONS.TEST dir
_C.TEST.ENABLED = True
# Batch size during test
_C.TEST.BATCH_SIZE = 1
# Run test procedure for JPEG images compressed with these quality factors
_C.TEST.QUALITIES = [10, 20, 40, 60, 80]
# Central crop region size.
# Each image in DATA.LOCATIONS.TEST is cropped to a square with this size
_C.TEST.REGION_SIZE = 512


# -----------------------------------------------------------------------------
# Logging setting
# -----------------------------------------------------------------------------
_C.LOGGING = CN()
# Log frequency in terms of iteration count
_C.LOGGING.LOG_EVERY = 100
# Parent log directory. Log directory for the run is LOGGING.DIR/TAG/
_C.LOGGING.DIR = "logs/"
# If `True`, use Wandb cloud service
_C.LOGGING.WANDB = True
# If `True`, save plot images
_C.LOGGING.PLOTS = True



def default_config():
    """Returns default config node"""
    return _C.clone()


def get_config(filename):
    """Reads config from file"""
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
