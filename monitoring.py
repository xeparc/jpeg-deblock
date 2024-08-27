import torchvision
import os
class TrainingMonitor:

    def __init__(self, logger, wandb=None):
        self.logger = logger

    def log(self, level, msg):
        self.logger.log(level, msg)

    def add_scalar(self, name, value):
        pass

    def add_histogram(self, name, bins, values):
        pass

    def add_image(self, name, image):
        os.makedirs("samples/", exist_ok=True)
        torchvision.io.write_png(image, f"samples/{name}.png")


class NullMonitor:

    def log(self, msg):
        pass

    def add_scalar(self, name, value):
        pass