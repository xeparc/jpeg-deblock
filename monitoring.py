import os
import numpy as np
import torchvision
import wandb


class TrainingMonitor:

    def __init__(self, logger, wandb=False):
        self.logger = logger
        self.wandb = wandb
        self._step = 0

    def log(self, level, msg):
        self.logger.log(level, msg)

    def add_scalar(self, name, value):
        if self.wandb:
            wandb.log({name: value}, step=self._step, commit=False)

    def add_histogram(self, name, bins, values):
        if self.wandb:
            np_hist = (np.asarray(values), np.asarray(bins))
            wandb.log({name: wandb.Histogram(np_histogram=np_hist)},
                      step=self._step, commit=False)

    def add_image(self, name, image):
        os.makedirs("samples/", exist_ok=True)
        torchvision.io.write_png(image, f"samples/{name}.png")
        if self.wandb:
            data = image.permute(1,2,0).numpy()
            wandb.log({name: wandb.Image(data, mode="RGB", caption=name)},
                      step=self._step, commit=False)

    def step(self):
        self._step += 1


class NullMonitor:

    def log(self, msg):
        pass

    def add_scalar(self, name, value):
        pass