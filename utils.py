import os
import numpy as np


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
        if self.count == 0:
            self._mean = np.zeros(x.shape, dtype=np.float32)
            self._var  = np.zeros(x.shape, dtype=np.float32)
        else:
            assert x.shape == self.mean.shape
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
        return self._mean

    @property
    def std(self):
        if self.count < 2:
            return None
        return np.sqrt(self._var / self.count)

    @property
    def var(self):
        if self.count < 2:
            return None
        return self._var / self.count


class CombinedLoss:

    def __init__(self, criterion, gamma, luma_weight, chroma_weight):
        self.criterion = criterion
        self.gamma = gamma
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
        total_loss = self.gamma * spectral_loss + (1 - self.gamma) * chroma_loss

        return {
            "Y":        self.luma_weight * Y_loss,
            "Cb":       self.chroma_weight * Cb_loss,
            "Cr":       self.chroma_weight * Cr_loss,
            "spectral": spectral_loss,
            "chroma":   chroma_loss
        }

