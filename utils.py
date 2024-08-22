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

class RunningMeanStd:

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