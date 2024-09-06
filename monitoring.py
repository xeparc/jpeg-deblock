from collections import defaultdict
import logging
import os
import pickle
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb


class TrainingMonitor:

    def __init__(self, logger, wandb=False):
        self.logger = logger
        self.wandb = wandb
        self._step = 0
        self.scalars = defaultdict(list)
        self.scalars_steps = defaultdict(list)

    def log(self, level, msg):
        self.logger.log(level, msg)

    def add_scalar(self, mapping: dict):
        if self.wandb:
            wandb.log(mapping, step=self._step, commit=False)
        for k, v in mapping.items():
            self.scalars[k].append(v)
            self.scalars_steps[k].append(self._step)

    def add_histogram(self, name, bins, values):
        if self.wandb:
            np_hist = (np.asarray(values), np.asarray(bins))
            wandb.log({name: wandb.Histogram(np_histogram=np_hist)},
                      step=self._step, commit=False)
        self.histograms[name].append((values, bins))

    def add_image(self, name, image):
        os.makedirs("samples/", exist_ok=True)
        torchvision.io.write_png(image, f"samples/{name}.png")
        C = image.shape[0]
        if self.wandb:
            data = image.permute(1,2,0).cpu().numpy()
            mode = "RGB" if C == 3 else "L"
            wandb.log({name: wandb.Image(data, mode=mode, caption=name)},
                      step=self._step, commit=False)

    def log_grad_norms(self, named_parameters: Mapping):
        for name, param in named_parameters:
            if param.grad is not None:
                norm = torch.linalg.norm(param.grad).item()
                std = param.grad.std().item()
                msg = f"\t[{name}]: grad norm = {norm}"
                self.logger.log(logging.DEBUG, msg)
                #
                self.scalars[f"grads/norm/{name}"].append(norm)
                self.scalars_steps[f"grads/norm/{name}"].append(self._step)
                #
                self.scalars[f"grads/std/{name}"].append(std)
                self.scalars_steps[f"grads/std/{name}"].append(self._step)

    def log_params(self, named_parameters: Mapping):
        for name, param in named_parameters:
            u = param.mean().item()
            s = param.std().item()
            msg  = f"\t[{name}]: parameter = {u} ± {s}"
            self.logger.log(logging.DEBUG, msg)
            #
            self.scalars[f"params/std/{name}"].append(s)
            self.scalars_steps[f"params/std/{name}"].append(self._step)
            #
            self.scalars[f"params/mean/{name}"].append(u)
            self.scalars_steps[f"params/mean/{name}"].append(self._step)

    def log_param_updates(self, old: Optional[Mapping], new: Optional[Mapping]):
        if old is None or new is None:
            return
        old = dict(old)
        new = dict(new)
        for name, new_param in new.items():
            if name not in old:
                continue
            diff = new_param.detach().cpu() - old[name].detach().cpu()
            u = diff.mean().item()
            s = diff.std().item()
            n  = torch.linalg.norm(diff).item() / (torch.linalg.norm(new_param).item() + 1e-6)
            msg = f"\t[{name}]: update = {u} ± {s}, relative update norm = {n}"
            self.logger.log(logging.DEBUG, msg)
            #
            self.scalars[f"updates/{name}"].append(n)
            self.scalars_steps[f"updates/{name}"].append(self._step)

    def save_state(self, savepath):
        savedir = os.path.dirname(savepath)
        os.makedirs(savedir, exist_ok=True)
        with open(savepath, mode="wb") as f:
            state = {
                "scalars": self.scalars,
                "scalars_steps": self.scalars_steps,
                "step": self._step
            }
            pickle.dump(state, f)

    def load_state(self, path):
        with open(path, mode="rb") as f:
            state = pickle.load(f)
            self._step = state["step"]
            self.scalars = state["scalars"]
            self.scalars_steps = state["scalars_steps"]

    def plot_scalars(self, savedir):
        for key in self.scalars:
            parts, name = os.path.split(key)
            if parts:
                os.makedirs(os.path.join(savedir, parts), exist_ok=True)
            xs = self.scalars_steps[key]
            ys = self.scalars[key]
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(xs, ys, linewidth=1)
            ax.set_title(name)
            ax.set_xlabel("iteration")
            ax.set_ylabel("value")
            fig.savefig(os.path.join(savedir, parts, name + ".png"), dpi=160)
            plt.close(fig)
        plt.close("all")

    def step(self):
        self._step += 1

    def get_step(self):
        return self._step


class NullMonitor:

    def log(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass