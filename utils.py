import logging
import os
import numpy as np
import torch
import yacs

from models import *

IMAGE_EXTENSIONS = "jpg jpeg bmp png tif tiff".split()
_VALID_TYPES = {tuple, list, str, int, float, bool}

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

class CombinedLoss:

    def __init__(self, criterion, luma_weight, chroma_weight, alpha, beta):
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta
        self.luma_weight = luma_weight
        self.chroma_weight = chroma_weight

    def __call__(self, predictions: dict, targets: dict):
        Y_loss =  self.criterion(predictions["Y"], targets["Y"])
        Cb_loss = self.criterion(predictions["Cb"], targets["Cb"])
        Cr_loss = self.criterion(predictions["Cr"], targets["Cr"])

        spectral_loss = (self.luma_weight * Y_loss +
                         self.chroma_weight * Cb_loss +
                         self.chroma_weight * Cr_loss)
        chroma_loss = self.criterion(predictions["final"], targets["final"])
        total_loss = self.alpha * spectral_loss + self.beta * chroma_loss

        return {
            "Y":        self.luma_weight * Y_loss,
            "Cb":       self.chroma_weight * Cb_loss,
            "Cr":       self.chroma_weight * Cr_loss,
            "spectral": spectral_loss,
            "chroma":   chroma_loss,
            "total":    total_loss
        }



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


def charbonnier_loss(input, target, reduction="mean", eps=1e-3):
    l = torch.sqrt((input - target) ** 2 + eps)
    if reduction == "mean":
        return torch.mean(l)
    elif reduction == "sum":
        return torch.sum(l)
    elif reduction == "none":
        return l
    else:
        raise ValueError


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


def collect_inputs(model, batch):
    if isinstance(model, RRDBNet):
        return dict(x=batch["lq_ycc"])
    elif isinstance(model, PrismLumaS4):
        return dict(y=batch["lq_y"], dct_y=batch["lq_dct_y"])
    else:
        raise NotImplementedError


def collect_target(model, batch):
    if isinstance(model, RRDBNet):
        return batch["hq_ycc"]
    elif isinstance(model, PrismLumaS4):
        return batch["hq_y"]


def get_alloc_memory(config):
    if config.TRAIN.DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / (2 ** 20)
    elif config.TRAIN.DEVICE == "mps":
        return torch.mps.current_allocated_memory() / (2 ** 20)
    else:
        return 0.0
