from typing import *
import torch
import torch.nn as nn

from .perceptual import Q1Net


class PerceptualLoss(nn.Module):

    def __init__(self,
                 layers: nn.Sequential,
                 indices: Iterable[int],
                 weights: Iterable[float],
                 norm: str = "l2",
                 input_transform: Optional[nn.Module] = None
    ):
        super().__init__()
        assert isinstance(layers, nn.Sequential)

        # Disable gradients
        self.layers = layers
        self.layers.requires_grad_(False)

        self.indices = set(indices)
        self.weights = list(weights)
        self.max_idx = max(self.indices)
        assert len(self.indices) == len(self.weights)

        if norm == "l2":
            self.criterion = nn.MSELoss()
        elif norm == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError(norm)

        if input_transform is None:
            self.input_transform = nn.Identity()
        else:
            self.input_transform = input_transform

    def forward(self, x, target):
        x       = self.input_transform(x)
        target  = self.input_transform(target)

        x_features = []
        for i in range(self.max_idx):
            x = self.layers[i](x)
            if i in self.indices:
                x_features.append(x)

        with torch.no_grad():
            target_features = []
            for i in range(self.max_idx):
                target = self.layers[i](target)
                if i in self.indices:
                    target_features.append(target)

        loss = 0.0
        for xf, tf, w in zip(x_features, target_features, self.weights):
            loss += w * self.criterion(xf, tf)

        return loss


class Q1PerceptualLoss:

    def __init__(self, params_path: str, indices=(2,3,5),
                 weights=(0.3,0.3,0.4), device="cpu"):
        q1net = Q1Net(in_channels=3)
        state = torch.load(params_path, weights_only=True, map_location="cpu")
        q1net.load_state_dict(state)
        q1net.requires_grad_(False)
        self.ploss = PerceptualLoss(q1net.features, indices, weights, norm="l1")
        self.ploss = self.ploss.to(device=device)

    def __call__(self, x, target):
        return self.ploss(x, target)


class MixedQ1MSELoss:

    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha
        self.ploss = Q1PerceptualLoss(**kwargs)

    def __call__(self, x, target):
        mse = nn.functional.mse_loss(x, target)
        perceptual = self.ploss(x, target)
        return self.alpha * mse + (1.0 - self.alpha) * perceptual


class CharbonnierLoss:

    def __call__(self, x, target, reduction="mean", eps=1e-3):
        l = torch.sqrt((x - target) ** 2 + eps)
        if reduction == "mean":
            return torch.mean(l)
        elif reduction == "sum":
            return torch.sum(l)
        elif reduction == "none":
            return l
        else:
            raise NotImplementedError("Unknown reduction: " + str(reduction))