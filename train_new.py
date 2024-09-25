"""Script for training CNN for JPEG Quality Assesment"""

import argparse
import datetime
import logging
import os
import time
from typing import *

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from yacs.config import CfgNode
import wandb

from config import get_config
from builders import *
from monitoring import TrainingMonitor
from utils import (
    collect_inputs,
    collect_target,
    save_checkpoint,
    load_checkpoint,
    yacs_to_dict,
    clip_gradients,
    get_alloc_memory
)


def train(
        config:         CfgNode,
        model:          torch.nn.Module,
        dataloader:     torch.utils.data.DataLoader,
        optimizer:      torch.optim.Optimizer,
        lr_scheduler:   torch.optim.lr_scheduler.LRScheduler,
        monitor:        TrainingMonitor,
        start_iter:     int,
        max_iters:      int
):

    current_iter =          start_iter
    device =                config.TRAIN.DEVICE
    accum =                 config.TRAIN.ACCUMULATE_GRADS
    clip_grad_method =      config.TRAIN.CLIP_GRAD_METHOD
    max_norm =              config.TRAIN.CLIP_GRAD
    checkpoint_every =      config.TRAIN.CHECKPOINT_EVERY
    log_every =             config.LOGGING.LOG_EVERY
    total_iters =           config.TRAIN.NUM_ITERATIONS

    dataiter = iter(dataloader)
    batch = next(dataiter)
    while current_iter < max_iters:
        tic = time.time()
        # Advance iteration number
        current_iter += 1
        monitor.step()

        # Zero gradients
        optimizer.zero_grad()

        # Collect input arguments to `model` & target
        inputs = collect_inputs(config, model, batch)
        target = collect_target(config, model, batch)

        # Transfer inputs & target to device
        inputs = {k: x.to(device=device) for k, x in inputs.items()}
        target = target.to(device=device)

        # Run forward pass (asynchronous)
        preds = model(**inputs)

        # Calculate loss
        loss = torch.nn.functional.mse_loss(preds, target) / accum

        # Run backward pass
        loss.backward()

        # Fetch next datapoint, while backward pass runs asynchronously on device
        try:
            batch = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            batch = next(dataiter)

        # Update parameters
        if current_iter % accum == 0:
            # Clip gradients
            clip_gradients(model, max_norm, clip_grad_method)
            # We're about to log. Save old paramters
            if current_iter % log_every == 0:
                old_params = {n: p.detach().clone() for n, p in model.named_parameters()}
            else:
                old_params = None
            optimizer.step()
            lr_scheduler.step()

        # Syncronize
        # if device == "mps":
        #     torch.mps.synchronize()
        # elif device == "cuda":
        #     torch.cuda.synchronize()

        # Track stats
        loss_ = accum * loss.detach().item()
        psnr_ = -10.0 * np.log10(loss_)
        lr_   = lr_scheduler.get_last_lr()[0]
        batch_time = time.time() - tic
        eta   = int(batch_time * (total_iters - current_iter))
        monitor.add_scalar({"loss": loss_, "lr": lr_, "psnr": psnr_})

        # Log stats
        if current_iter % log_every == 0:
            monitor.log(
                logging.INFO,
                f"Train: [{current_iter:>6}/{total_iters:>6}]\t"
                f"time: {batch_time:.2f}\t"
                f"eta: {datetime.timedelta(seconds=eta)}\t"
                f"loss: {loss_:.4f}\t"
                f"psnr: {psnr_:.5f}\t"
                f"lr: {lr_:.6f}\t"
                f"memory: {get_alloc_memory(config):.0f}MB"
            )
            # Log gradient norms
            monitor.log_grad_norms(model.named_parameters())
            # Log parameters
            monitor.log_params(model.named_parameters())
            # Log relative parameter updates
            monitor.log_param_updates(old_params, model.named_parameters())
            del old_params

        # Checkpoint model
        if current_iter % checkpoint_every == 0:
            savestate = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "iteration": current_iter,
            }
            save_checkpoint(savestate, config, monitor)


@torch.no_grad()
def validate_v2(
    config:     CfgNode,
    model:      torch.nn.Module,
    quality:    Union[int, List[int]],
    monitor:    TrainingMonitor
):

    tic = time.time()
    val_losses = {}
    device = config.TRAIN.DEVICE

    if isinstance(quality, int):
        quality = [quality]

    for q in quality:
        # Init dataloader with input image quality = `q`
        dataloader = build_dataloader(config, kind="val", quality=q)
        # Iterate trough validation images
        for batch in dataloader:
            inputs = collect_inputs(config, model, batch)
            target = collect_target(config, model, batch)

            # Transfer inputs & target to device
            inputs = {k: x.to(device=device, non_blocking=True) for k, x in inputs.items()}
            target = target.to(device=device, non_blocking=True)

            preds = model(**inputs)
            loss = torch.nn.functional.mse_loss(preds, target)
            val_losses.setdefault(q, []).append(loss.item())

    current_iter = monitor.get_step()
    total_iters = config.TRAIN.NUM_ITERATIONS

    for q, loss_values in val_losses.items():
        loss_mean = np.mean(loss_values)
        loss_std  = np.std(loss_values)
        psnr = -10.0 * np.log10(loss_mean)
        monitor.log(
            logging.INFO,
            f"Validate: [{current_iter:>6}/{total_iters:>6}]\t"
            f"loss, q={q}: {loss_mean:.4f} ± {loss_std:.4f}\t"
            f"psnr, q={q}: {psnr:.5f}"
        )
        monitor.add_scalar({f"validation/loss-q={q}": loss_mean,
                            f"validation/confidence-q={q}": loss_std,
                            f"validation/psnr-q={q}": psnr})

    monitor.log(
        logging.INFO,
        f"Validate: [{current_iter:>6}/{total_iters:>6}]\t"
        f"time: {(time.time() - tic):.2f}"
    )
    return


@torch.no_grad()
def test_samples_v2(
        config:     CfgNode,
        model:      torch.nn.Module,
        quality:    Union[int, List[int]],
        monitor:    TrainingMonitor
):
    current_iter = monitor.get_step()
    total_iters  = config.TRAIN.NUM_ITERATIONS
    device       = config.TRAIN.DEVICE

    # Create output directory. `os.makedirs()` is with `exist_ok=False`,
    # because this directory should not exist. It it exists, this function
    # will probably overwrite images in it and that's not what we want
    savedir = os.path.join(config.LOGGING.DIR, config.TAG, "testsamples", str(current_iter))
    os.makedirs(savedir, exist_ok=False)

    to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)

    if isinstance(quality, int):
        quality = [quality]

    tic = time.time()
    for q in quality:
        # initialize dataloader with quality = `q`
        dataloader = build_dataloader(config, kind="test", quality=q)
        # Iterate trough all test images, encoded with quality = `q`
        for batch in dataloader:
            # Collect & transfer inputs to device
            inputs = collect_inputs(config, model, batch)
            inputs = {k: x.to(device=device, non_blocking=True) for k, x in inputs.items()}
            # Make predictions
            preds = model(**inputs)
            # Save images
            filepaths = batch["filepath"]
            for path, pred in zip(filepaths, preds):
                name = os.path.basename(path)
                # Strip extension from source filename
                name = '.'.join(name.split('.')[:-1])
                savepath = os.path.join(savedir, name + f"-q={q}.png")
                # Convert image from float32 to uint8 and save it as PNG
                img = to_uint8(torch.clip(pred.cpu(), 0.0, 1.0))
                torchvision.io.write_png(img, savepath)
    t = time.time() - tic

    monitor.log(
        logging.INFO,
        f"Test samples: [{current_iter:>6}/{total_iters:>6}]\t"
        f"time: {t:.2f}\t"
    )
    return


def train_validate_loop(
        config:         CfgNode,
        model:          torch.nn.Module,
        train_loader:   torch.utils.data.DataLoader,
        optimizer:      torch.optim.Optimizer,
        lr_scheduler:   torch.optim.lr_scheduler.LRScheduler,
        monitor:        TrainingMonitor
):

    val_every       = config.VALIDATION.EVERY
    max_iters       = config.TRAIN.NUM_ITERATIONS
    test_qualities  = config.TEST.QUALITIES
    val_qualities   = config.VALIDATION.QUALITIES

    plots_dir = os.path.join(config.LOGGING.DIR, config.TAG, "plots")
    monitor_path = os.path.join(config.LOGGING.DIR, config.TAG, "monitor.pickle")

    for i in range(0, max_iters, val_every):
        train(config, model, train_loader, optimizer, lr_scheduler,
              monitor, start_iter=i, max_iters=min(max_iters, i+val_every))
        # Validate
        validate_v2(config, model, val_qualities, monitor)
        # Test samples
        if config.TEST.ENABLED:
            test_samples_v2(config, model, test_qualities, monitor)
        # Plots
        if config.LOGGING.PLOTS:
            monitor.plot_scalars(plots_dir)
        monitor.save_state(monitor_path)
        # Repeat...


def main(cfg: CfgNode):

    # Init device
    device = torch.device(cfg.TRAIN.DEVICE)

    # Init logger
    logger          = build_logger(cfg)
    logger.log(logging.INFO, "Logger initialized!")

    # Init WANDB (is config flag for it is on)
    if cfg.LOGGING.WANDB:
        wandb.init(project=cfg.TAG, config=yacs_to_dict(cfg))

    # Init monitor
    monitor         = TrainingMonitor(logger, cfg.LOGGING.WANDB)

    # Init dataloaders
    train_loader    = build_dataloader(cfg, "train")
    logger.log(logging.INFO, "Train dataloader initialized!")

    # Init models
    model           = build_model(cfg).to(device)
    if cfg.LOGGING.WANDB:
        wandb.watch(model, log="all", log_freq=100, log_graph=False)

    # Init optimizer & scheduler
    optim           = build_optimizer(cfg, model.parameters())
    lr_scheduler    = build_lr_scheduler(cfg, optim)

    # Check if checkpoint directory already exists. It shouldn't !
    if cfg.TRAIN.CHECKPOINT_EVERY > 0:
        if os.path.exists(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, cfg.TAG)):
            raise(OSError("Checkpoint directory already exists!"))

    # Optionally load checkpoint
    if cfg.TRAIN.RESUME:
        state = {
            "model": model,
            "optimizer": optim,
            "lr_scheduler": lr_scheduler,
            "iteration": cfg.TRAIN.START_ITERATION,
        }
        cfg = load_checkpoint(state, cfg, monitor)
        model = state["model"]
        optim = state["optimizer"]
        lr_scheduler = state["lr_scheduler"]

    # Jump to training loop
    train_validate_loop(
        config=             cfg,
        model=              model,
        train_loader=       train_loader,
        optimizer=          optim,
        lr_scheduler=       lr_scheduler,
        monitor=            monitor
    )

    # Save model & optimizer
    modelpath = os.path.join(cfg.LOGGING.DIR, cfg.TAG, "model.pth")
    optimpath = os.path.join(cfg.LOGGING.DIR, cfg.TAG, "optimizer.pth")
    torch.save(model.state_dict(), modelpath)
    torch.save(optim.state_dict(), optimpath)

    # Exit wandb
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