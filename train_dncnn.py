"""Script for training ARCNN"""

import argparse
import datetime
import logging
import os
import time
from typing import List

import numpy as np
import torch
import torchvision
import wandb
from yacs.config import CfgNode

from builders import *
from config import get_config
from models.dncnn import DnCNN, DnCNN3
from monitoring import TrainingMonitor
from utils import (
    save_checkpoint,
    clip_gradients,
    get_alloc_memory,
    yacs_to_dict
)


def train(
        config: CfgNode,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        monitor,
        start_iter: int,
        max_iters: int
    ):

    current_iter =          start_iter
    accum =                 config.TRAIN.ACCUMULATE_GRADS
    batch_size =            config.TRAIN.BATCH_SIZE
    clip_grad_method =      config.TRAIN.CLIP_GRAD_METHOD
    max_norm =              config.TRAIN.CLIP_GRAD
    checkpoint_every =      config.TRAIN.CHECKPOINT_EVERY
    log_every =             config.LOGGING.LOG_EVERY
    total_iters =           config.TRAIN.NUM_ITERATIONS

    model.train()

    while current_iter < max_iters:
        for batch in dataloader:
            monitor.step()
            current_iter += 1
            if current_iter > max_iters:
                break
            optimizer.zero_grad()

            # Run forward pass, make predictions
            tic = time.time()
            # preds = model(batch["lq_rgb"])
            # targets = batch["hq_rgb"]
            preds = model(batch["lq_y"])
            targets = batch["hq_y"]

            # Calculate loss
            loss = torch.nn.functional.mse_loss(preds, targets, reduction="sum") / (accum * batch_size)
            loss.backward()

            # Update parameters
            if current_iter % accum == 0:
                # Clip gradients
                clip_gradients(model, max_norm, clip_grad_method)
                # We're about to log. Save old paramters
                if current_iter % log_every == 0:
                    old_params = {n: p.detach().cpu() for n, p in model.named_parameters()}
                else:
                    old_params = None
                optimizer.step()
                lr_scheduler.step()

            # Track stats
            batch_time = time.time() - tic
            loss_ = accum * loss.detach().cpu().item()
            lr_   = lr_scheduler.get_last_lr()[0]
            eta   = int(batch_time * (total_iters - current_iter))
            psnr_  = -10.0 * np.log10(loss_ + 1e-6)
            monitor.add_scalar({"loss/total": loss_, "lr": lr_, "psnr": psnr_})

            # Log stats
            if current_iter % log_every == 0:
                monitor.log(
                    logging.INFO,
                    f"Train: [{current_iter:>6}/{total_iters:>6}]\t"
                    f"time: {batch_time:.2f}\t"
                    f"eta: {datetime.timedelta(seconds=eta)}\t"
                    f"loss: {loss_:.4f}\t"
                    f"lr: {lr_:.6f}\t"
                    f"memory: {get_alloc_memory(config):.0f}MB"
                )
                # Log gradient norms
                monitor.log_grad_norms(model.named_parameters())
                # Log parameters
                monitor.log_params(model.named_parameters())
                # Log relative parameter updates
                monitor.log_param_updates(old_params, model.named_parameters())

            # Checkpoint model
            if current_iter % checkpoint_every == 0:
                savestate = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "iteration": current_iter,
                    "psnr": -1
                }
                save_checkpoint(savestate, config, monitor)


@torch.no_grad()
def validate(config: CfgNode, model: SpectralModel,
             dataloader: torch.utils.data.DataLoader, monitor):

    monitor.log(logging.INFO, "validate()...")
    model.eval()

    tic = time.time()
    total_loss = []
    for batch in dataloader:
        # targets = batch["hq_rgb"]
        # preds = model(batch["lq_rgb"])
        targets = batch["hq_y"]
        preds = model(batch["lq_y"])
        loss = torch.nn.functional.mse_loss(preds, targets) / len(targets)
        total_loss.append(loss.item())

    monitor.add_scalar({"validation/loss": np.mean(total_loss),
                        "validation/confidence": np.std(total_loss),
                        "validation/psnr": -10.0 * np.log10(np.mean(total_loss) + 1e-6)
                        })

    t = int(time.time() - tic)
    monitor.log(logging.INFO,
                f"validate() took {datetime.timedelta(seconds=t)}")
    monitor.step()


@torch.no_grad()
def test_samples(
        config: CfgNode,
        model: torch.nn.Module,
        filepaths: List[str],
        monitor
    ):
    monitor.log(logging.INFO, "test_samples()...")

    model.eval()
    device = torch.device(config.TRAIN.DEVICE)
    to_float = torchvision.transforms.ConvertImageDtype(torch.float32)
    to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)

    tic = time.time()
    for path in filepaths:
        name = '.'.join(os.path.basename(path).split('.')[:-1])
        img = to_float(torchvision.io.read_image(path))
        img = img.unsqueeze(0).to(device)
        pred = model(img)
        pred = torch.clip(pred[0].cpu(), 0.0, 1.0)
        pred = to_uint8(pred)
        monitor.add_image(name, pred)

    t = int(time.time() - tic)
    monitor.log(logging.INFO, f"test_samples() took {datetime.timedelta(seconds=t)}")


def train_validate_loop(config, model, train_loader, val_loader, optimizer,
                        lr_scheduler, monitor):

    val_every =         config.VALIDATION.EVERY
    max_iters =         config.TRAIN.NUM_ITERATIONS

    samples_dir = "data/Live1-Classic5/live1/gray/qf_10/"

    filepaths = [x.path for x in os.scandir(samples_dir) if x.name.endswith("jpg")]

    for i in range(0, max_iters, val_every):
        train(config, model, train_loader, optimizer, lr_scheduler,
              monitor, start_iter=i, max_iters=i+val_every)
        # Validate
        validate(config, model, val_loader, monitor)
        # Test samples
        test_samples(config, model, filepaths, monitor)
        # Repeat...


def main(cfg):

    # Init device
    device = torch.device(cfg.TRAIN.DEVICE)

    # Init WANDB (is config flag for it is on)
    if cfg.LOGGING.WANDB:
        wandb.init(project=cfg.TAG, config=yacs_to_dict(cfg))

    # Init dataloaders
    train_loader    = build_dataloader(cfg, "train")
    val_loader      = build_dataloader(cfg, "val")

    # Init models
    model           = DnCNN3(in_channels=1).to(device)
    optim           = build_optimizer(cfg, model.parameters())
    if cfg.LOGGING.WANDB:
        wandb.watch(model, log="all", log_freq=100, log_graph=False)

    # Init optimizer
    lr_scheduler    = build_lr_scheduler(cfg, optim)
    logger          = build_logger(cfg)
    monitor         = TrainingMonitor(logger, cfg.LOGGING.WANDB)

    # Create checkpoint directory
    if cfg.TRAIN.CHECKPOINT_EVERY > 0:
        os.makedirs(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, cfg.MODEL.NAME),
                    exist_ok=True)

    # Jump to training loop
    train_validate_loop(
        config=             cfg,
        model=              model,
        train_loader=       train_loader,
        val_loader=         val_loader,
        optimizer=          optim,
        lr_scheduler=       lr_scheduler,
        monitor=            monitor
    )

    if cfg.LOGGING.WANDB:
        wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    args = parser.parse_args()

    cfg = get_config(args.config)
    cfg_str = cfg.dump()
    cfg_savepath = os.path.join(cfg.LOGGING.DIR, cfg.TAG, "config.yaml")
    os.makedirs(os.path.dirname(cfg_savepath), exist_ok=True)
    with open(cfg_savepath, mode="w") as f:
        cfg.dump(stream=f)

    main(cfg)