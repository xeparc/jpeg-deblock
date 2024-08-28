import logging
import os
import numpy as np
import torch
import yacs

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



def load_checkpoint(state, config, logger):
    logger.error(f"========= > Resuming from {config.MODEL.RESUME}")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=True)
    iter = checkpoint["iteration"]
    for k, v in checkpoint.items():
        if k in state and isinstance(state[k], torch.nn.Module):
            state[k].load_state_dict(checkpoint[k])
            logger.error(f"=== > loaded successfully \"{k}\" (iter {iter})")
        elif k in state:
            state[k] = checkpoint[k]
        else:
            logger.error(f"=== > failed to load \"{k}\" (iter {iter})")
    config.defrost()
    config.TRAIN.START_ITERATION = checkpoint["iteration"] + 1


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

_VALID_TYPES = {tuple, list, str, int, float, bool}

def yacs_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, yacs.config.CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = yacs_to_dict(v, key_list + [k])
        return cfg_dict
