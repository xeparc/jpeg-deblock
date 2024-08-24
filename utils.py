import os
import numpy as np
import torch

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



def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"========= > Resuming from {config.MODEL.RESUME}")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    config.defrost()
    config.TRAIN.START_ITERATION = checkpoint["iteration"] + 1
    logger.info(f"========= > loaded successfully '{config.MODEL.RESUME}' (iter {checkpoint['iteration']})")
    psnr = checkpoint["psnr"]
    del checkpoint
    torch.cuda.empty_cache()
    return psnr


def save_checkpoint(state, config, iter, logger):
    state["iteration"] = iter
    savepath = os.path.join(
        config.TRAIN.CHECKPOINT_DIR,
        config.TAG,
        f'checkpoint_{iter}.pth'
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    logger.info(f"{savepath} saving......")
    torch.save(state, savepath)
    logger.info(f"{savepath} saved !!!")