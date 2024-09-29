import logging
import os
import numpy as np
import torch
import yacs

from models import *

IMAGE_EXTENSIONS = "jpg jpeg bmp png tif tiff".split()


def is_image(filepath):
    name = os.path.basename(filepath)
    try:
        extension = name.split('.')[-1]
        return extension in IMAGE_EXTENSIONS
    except:
        return False

class RunningStats:

    def __init__(self):
        self.count = 0

    def update(self, x):
        x = np.asarray(x)
        if self.count == 0:
            self._mean = np.zeros(x.shape, dtype=np.float32)
            self._var  = np.zeros(x.shape, dtype=np.float32)
        else:
            assert x.shape == self._mean.shape
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        delta2 = x - self._mean
        self._var += delta * delta2

    def reset(self):
        self.count = 0

    @property
    def mean(self):
        if self.count == 0:
            return None
        return float(self._mean) if self._mean.size == 1 else self._mean

    @property
    def std(self):
        if self.count < 2:
            return 0.0
        res = np.sqrt(self._var / self.count)
        return float(res) if res.size == 1 else res

    @property
    def var(self):
        if self.count < 2:
            return 0.0
        res = self._var / self.count
        return float(res) if res.size == 1 else res


def load_checkpoint(state, config, monitor):
    monitor.log(logging.INFO, f"=== > Resuming from {config.TRAIN.RESUME}")
    checkpoint = torch.load(config.TRAIN.RESUME, map_location='cpu')
    iter = checkpoint["iteration"]

    for k in checkpoint.keys():
        if k not in state:
            continue
        if hasattr(state[k], "load_state_dict"):
            state[k].load_state_dict(checkpoint[k])
        else:
            state[k] = checkpoint[k]
        monitor.log(logging.INFO, f"=== > loaded successfully \"{k}\" (iter {iter})")

    config.defrost()
    config.TRAIN.START_ITERATION = checkpoint["iteration"] + 1
    return config


def save_checkpoint(state, config, monitor):
    i = state["iteration"]
    savepath = os.path.join(
        config.TRAIN.CHECKPOINT_DIR,
        config.TAG,
        f'checkpoint_{i}.pth'
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    monitor.log(logging.INFO, f"{savepath} saving......")
    torch.save(state, savepath)
    monitor.log(logging.INFO, f"{savepath} saved !!!")


def yacs_to_dict(cfg_node):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, yacs.config.CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = yacs_to_dict(v)
        return cfg_dict


def clip_gradients(model, max_norm: float, how: str):

    if max_norm <= 0.0:
        return {}

    model_name = type(model).__name__
    result = {}

    if how == "total":
        parameters = model.parameters()
        g = torch.nn.utils.clip_grad_norm_(parameters, max_norm,
                                           error_if_nonfinite=True)
        result[model_name] = g.item()
    elif how == "param":
        for name, param in model.named_parameters():
            g = torch.nn.utils.clip_grad_norm_([param], max_norm,
                                               error_if_nonfinite=True)
            result[name] = g.item()
    return result


def collect_inputs(config, model, batch):
    if isinstance(model, (MobileNetQA, Q1Net)):
        k = config.MODEL.INPUTS[0]
        result = dict(x=batch[k])
    elif isinstance(model, RRDBNet):
        result = dict(x=batch["lq_ycc"])
    elif isinstance(model, Prism):
        result = dict(y=batch["lq_y"], dct_y=batch["lq_dct_y"])
    elif isinstance(model, FlareLuma):
        result = dict(y=batch["lq_y"], dct_y=batch["lq_dct_y"], qt=batch["qt_y"])
    elif isinstance(model, FlareChroma):
        result = dict(
            y=      batch["hq_y"],
            cb=     batch["lq_cb"],
            cr=     batch["lq_cr"],
            dct_cb= batch["lq_dct_cb"],
            dct_cr= batch["lq_dct_cr"],
            qt_c=   batch["qt_c"]
        )
    elif isinstance(model, Flare):
        result = dict(
            y=      batch["hq_y"] if model.use_hq_luma else batch["lq_y"],
            cb=     batch["lq_cb"],
            cr=     batch["lq_cr"],
            dct_y=  batch["lq_dct_y"],
            dct_cb= batch["lq_dct_cb"],
            dct_cr= batch["lq_dct_cr"],
            qt_y=   batch["qt_y"],
            qt_c=   batch["qt_c"]
        )
    else:
        raise NotImplementedError
    return result


def collect_target(config, model, batch):
    if isinstance(model, (MobileNetQA, Q1Net)):
        k = config.MODEL.TARGETS[0]
        # Normalize `quality` to [0,1]
        return batch[k] / 100
    if isinstance(model, RRDBNet):
        return batch["hq_ycc"]
    if isinstance(model, Prism):
        return batch["hq_y"]
    if isinstance(model, FlareLuma):
        return batch["hq_y"]
    if isinstance(model, FlareChroma):
        return torch.cat([batch["hq_cb"], batch["hq_cr"]], dim=1)
    if isinstance(model, Flare):
        return batch["hq_rgb"] if model.rgb_output else batch["hq_ycc"]
    raise NotImplementedError


def get_alloc_memory(config):
    if config.TRAIN.DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / (2 ** 20)
    elif config.TRAIN.DEVICE == "mps":
        return torch.mps.current_allocated_memory() / (2 ** 20)
    else:
        return 0.0
