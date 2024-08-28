import argparse
import datetime
import logging
import os
import time
from typing import List, Callable
import yaml

import numpy as np
from yacs.config import CfgNode
import wandb

from config import default_config, get_config
from builders import *
from inference import (
    predict_spectral,
    deartifact_jpeg
)
from jpegutils import SUBSAMPLE_FACTORS
from monitoring import TrainingMonitor, NullMonitor
from utils import (
    RunningStats,
    save_checkpoint,
    load_checkpoint,
    charbonnier_loss,
    is_image,
    yacs_to_dict
)



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
    if cfg.MODEL.SHARED_LUMA_CHROMA:
        spectral_luma   = build_spectral_model(cfg).to(device)
        spectral_chroma = build_spectral_model(cfg).to(device)
        optim           = build_optimizer(cfg, spectral_luma.parameters(),
                                               spectral_chroma.parameters())
        if cfg.LOGGING.WANDB:
            wandb.watch((spectral_luma, spectral_chroma), log="all",
                        log_freq=100, log_graph=True)
    else:
        spectral        = build_spectral_model(cfg).to(device)
        spectral_luma   = spectral
        spectral_chroma = spectral
        optim           = build_optimizer(cfg, spectral.parameters())
        if cfg.LOGGING.WANDB:
            wandb.watch(spectral, log="all", log_freq=100, log_graph=True)

    # Init optimizer
    lr_scheduler    = build_lr_scheduler(cfg, optim)
    logger          = build_logger(cfg)
    monitor         = TrainingMonitor(logger, cfg.LOGGING.WANDB)

    # Create checkpoint directory
    if cfg.TRAIN.CHECKPOINT_EVERY > 0:
        os.makedirs(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, cfg.MODEL.NAME),
                    exist_ok=True)

    # Jump to training loop
    train_validate_test_loop_spectral(
        config=             cfg,
        spectral_luma=      spectral_luma,
        spectral_chroma=    spectral_chroma,
        train_loader=       train_loader,
        val_loader=         val_loader,
        optimizer=          optim,
        lr_scheduler=       lr_scheduler,
        monitor=            monitor
    )

    if cfg.LOGGING.WANDB:
        wandb.finish()


def clip_gradients(model, max_norm: float, how: str):

    model_name = type(model).__name__

    if how == "total":
        parameters = model.parameters()
        g = torch.nn.utils.clip_grad_norm_(parameters, max_norm,
                                           error_if_nonfinite=True)
    elif how == "param":
        for name, param in model.named_parameters():
            g = torch.nn.utils.clip_grad_norm_([param], max_norm,
                                               error_if_nonfinite=True)

def log_grad_norms(model, monitor, current_iter: int, model_name: str = ''):
    pass
    # else:
    #     total_norm = 0.0
    #     for name, param in model.named_parameters():
    #         msg = f"total grad norm [{model_name}]: {g.cpu().item():.5f}"
    #     monitor.log(logging.DEBUG, msg)



def spectral_loss(config, prediction: dict, target: dict, monitor, **kwargs):
    output_type   = config.MODEL.SPECTRAL.OUTPUT_TRANSFORM
    loss_type     = config.LOSS.CRITERION
    luma_weight   = config.LOSS.LUMA_WEIGHT
    chroma_weight = config.LOSS.CHROMA_WEIGHT

    if loss_type == "mse":
        criterion = torch.nn.functional.mse_loss
    elif loss_type == "huber":
        criterion = torch.nn.functional.huber_loss
    elif loss_type == "charbonnier":
        criterion = charbonnier_loss

    if output_type == "identity":
        lossY  = criterion(prediction["dctY"], target["dctY"], **kwargs)
        lossCb = criterion(prediction["dctCb"], target["dctCb"], **kwargs)
        lossCr = criterion(prediction["dctCr"], target["dctCr"], **kwargs)
        # Track loss. Explicitly use mean(), because kwargs may contain
        # reduction="none"
        monitor.add_scalar("loss/dct-Y", lossY.detach().cpu().mean().item())
        monitor.add_scalar("loss/dct-Cb", lossCb.detach().cpu().mean().item())
        monitor.add_scalar("loss/dct-Cr", lossCr.detach().cpu().mean().item())
        return luma_weight * lossY + chroma_weight * (lossCb + lossCr)
    elif output_type == "ycc":
        loss = criterion(prediction["ycc"], target["ycc"], **kwargs)
        # Track loss. Explicitly use mean(), because kwargs may contain
        # reduction="none"
        monitor.add_scalar("loss/YCbCr", loss.detach().cpu().mean().item())
        return loss
    elif output_type == "rgb":
        loss = criterion(prediction["rgb"], target["rgb"], **kwargs)
        # Track loss. Explicitly use mean(), because kwargs may contain
        # reduction="none"
        monitor.add_scalar("loss/rgb", loss.detach().cpu().mean().item())
        return loss
    else:
        raise ValueError


@torch.no_grad()
def validate_spectral(
        config: CfgNode,
        spectral_luma: SpectralModel,
        spectral_chroma: SpectralModel,
        dataloader: torch.utils.data.DataLoader,
        monitor
    ):

    monitor.log(logging.INFO, "validate_spectral()...")

    subsample =             config.DATA.SUBSAMPLE
    out_transform =         config.MODEL.SPECTRAL.OUTPUT_TRANSFORM

    losses = {}
    psnrs  = {}
    nullmonitor = NullMonitor()

    tic = time.time()
    for batch in dataloader:
        quality = batch["quality"]
        # Run forward pass, make predictions
        preds = predict_spectral(config, spectral_luma, spectral_chroma, batch,
                                 out_transform, subsample)
        targets = {"dctY": batch["hq_dct_y"], "dctCb": batch["hq_dct_cb"],
                   "dctCr": batch["hq_dct_cr"]}
        if out_transform == "ycc":
            targets["ycc"] = batch["hq_ycc"]
        elif out_transform == "rgb":
            targets["rgb"] = batch["hq_rgb"]

        # Calculate loss
        loss = spectral_loss(config, preds, targets, nullmonitor, reduction="none")
        assert loss.ndim == 4
        loss = loss.mean(dim=(1,2,3)).cpu().numpy()
        for q, l in zip(quality, loss):
            # Aggregate over qualities with bin size = 10
            key = (q // 10) * 10
            losses.setdefault(key, []).append(l)
            psnrs.setdefault(key, []).append(-10.0 * np.log10(l - 1e-6))

    bins = sorted(losses.keys())
    mean_  = [np.mean(losses[k]) for k in bins]
    psnr_  = [np.mean(psnrs[k]) for k in bins]
    for i in range(len(bins)):
        k = bins[i]
        monitor.add_scalar(f"validation/loss-q=[{k},{k+10}]", mean_[i])
        monitor.add_scalar(f"validation/psnr-q=[{k},{k+10}]", psnr_[i])

    t = int(time.time() - tic)
    monitor.add_scalar("validation-time", t)
    monitor.log(logging.INFO,
                f"validate_spectral() took {datetime.timedelta(seconds=t)}")
    monitor.step()


@torch.no_grad()
def sample_test_spectral(
        config: CfgNode,
        spectral_luma: SpectralModel,
        spectral_chroma: SpectralModel,
        filepaths: List[str],
        monitor
    ):
    monitor.log(logging.INFO, "sample_test_spectral()...")

    tic = time.time()
    for path in filepaths:
        name = '.'.join(os.path.basename(path).split('.')[:-1])
        img = deartifact_jpeg(config, spectral_luma, spectral_chroma, path)
        monitor.add_image(name, img)

    t = int(time.time() - tic)
    monitor.log(logging.INFO,
                f"sample_test_spectral() took {datetime.timedelta(seconds=t)}")


def train_spectral(
        config: CfgNode,
        spectral_luma: SpectralModel,
        spectral_chroma: SpectralModel,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        monitor,
        start_iter: int,
        max_iters: int
    ):

    current_iter =          start_iter
    subsample =             config.DATA.SUBSAMPLE
    out_transform =         config.MODEL.SPECTRAL.OUTPUT_TRANSFORM
    accum =                 config.TRAIN.ACCUMULATE_GRADS
    clip_grad_method =      config.TRAIN.CLIP_GRAD_METHOD
    max_norm =              config.TRAIN.CLIP_GRAD
    checkpoint_every =      config.TRAIN.CHECKPOINT_EVERY
    shared_models =         config.MODEL.SHARED_LUMA_CHROMA
    log_every =             config.LOGGING.LOG_EVERY
    total_iters =           config.TRAIN.NUM_ITERATIONS

    spectral_luma.train()
    spectral_chroma.train()

    while current_iter < max_iters:
        for batch in dataloader:
            monitor.step()
            current_iter += 1
            if current_iter >= max_iters:
                break
            optimizer.zero_grad()

            # Collect targets
            targets = {"dctY": batch["hq_dct_y"], "dctCb": batch["hq_dct_cb"],
                       "dctCr": batch["hq_dct_cr"]}
            if out_transform == "ycc":
                targets["ycc"] = batch["hq_ycc"]
            elif out_transform == "rgb":
                targets["rgb"] = batch["hq_rgb"]

            # Run forward pass, make predictions
            tic = time.time()
            preds = predict_spectral(config, spectral_luma, spectral_chroma,
                                     batch, out_transform, subsample)

            # Calculate loss
            loss = spectral_loss(config, preds, targets, monitor) / accum
            loss.backward()

            # Update parameters
            if current_iter % accum == 0:
                # Clip gradients
                # clip_gradients(spectral_luma, max_norm, clip_grad_method)
                optimizer.step()
                lr_scheduler.step()

            # Track stats
            batch_time = time.time() - tic
            loss_ = accum * loss.detach().cpu().item()
            lr_   = lr_scheduler.get_last_lr()[0]
            eta   = int(batch_time * (total_iters - current_iter))

            monitor.add_scalar("loss/total", loss_)
            monitor.add_scalar("learning-rate", lr_)
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
                if shared_models:
                    log_grad_norms(spectral_luma, monitor, current_iter,
                                   "SpectralModel[Shared]")
                else:
                    log_grad_norms(spectral_luma, monitor, current_iter,
                                "SpectralModel[Luma]")
                    log_grad_norms(spectral_chroma, monitor, current_iter,
                                "SpectralModel[Chroma]")

                # TODO Track relative parameter updates

            # Checkpoint models
            if current_iter % checkpoint_every == 0:
                savestate = {
                    "spectral_luma": spectral_luma.state_dict(),
                    "spectral_chroma": spectral_chroma.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "iteration": current_iter,
                    "psnr": -1
                }
                save_checkpoint(savestate, config, monitor)


def train_validate_test_loop_spectral(
        config,
        spectral_luma,
        spectral_chroma,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        monitor
):

    val_every =         config.VALIDATION.EVERY
    max_iters =         config.TRAIN.NUM_ITERATIONS

    # Collect test samples filepaths
    test_filepaths = []
    for item in os.scandir(config.TEST.SAMPLES_DIR):
        if is_image(item.name):
            test_filepaths.append(item.path)

    for i in range(0, max_iters, val_every):
        train_spectral(config, spectral_luma, spectral_chroma, train_loader,
                       optimizer, lr_scheduler, monitor, start_iter=i,
                       max_iters=i+val_every)
        # Validate
        validate_spectral(config, spectral_luma, spectral_chroma, val_loader,
                          monitor)
        # Save sample images
        sample_test_spectral(config, spectral_luma, spectral_chroma,
                             test_filepaths, monitor)
        # Repeat...


def train_loop(
        config: CfgNode,
        spectral_net: torch.nn.Module,
        chroma_net: torch.nn.Module,
        criterion: Callable,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        logger
    ):

    spectral_net.train()
    chroma_net.train()
    optimizer.zero_grad()

    rgbout = config.MODEL.RGB_OUTPUT
    clip_grad = config.TRAIN.CLIP_GRAD > 0.0

    max_iters = config.TRAIN.NUM_ITERATIONS
    current_iter = 0
    subsample   = config.DATA.SUBSAMPLE
    batch_time_meter = RunningStats()
    total_loss_meter = RunningStats()
    spectral_loss_meter = RunningStats()
    chroma_loss_meter = RunningStats()
    spectral_grad_norm_meter = RunningStats()
    chroma_grad_norm_meter = RunningStats()

    while True:
        train_iterable = iter(train_loader)
        for datapoint in train_iterable:
            # Increment iteration number
            if current_iter >= max_iters:
                break
            current_iter += 1
            optimizer.zero_grad()

            tic = time.time()
            # Make predictions
            preds = predict_train(spectral_net, chroma_net, datapoint, subsample)
            targets = {
                "Y":        datapoint["hq_y"],
                "Cb":       datapoint["hq_cb"],
                "Cr":       datapoint["hq_cr"],
                "final":    datapoint["hq_rgb"] if rgbout else datapoint["hq_ycc"]
            }

            # Compute loss
            loss = criterion(preds, targets)
            loss["total"].backward()

            # # Collect gradients
            grad_norms_ = {}
            # for name, param in spectral_net.named_parameters():
            #     if hasattr(param, "grad"):
            #         grad_norms_[name] = torch.linalg.norm(param.clone().detach().cpu()).item()

            # Collect old parameter values
            param_values_ = {}
            for name, param in spectral_net.named_parameters():
                if hasattr(param, "grad"):
                    param_values_[name] = param.detach().cpu().clone()

            # Clip gradients (per parameter)
            if clip_grad:
                with torch.no_grad():
                    for name, param in spectral_net.named_parameters():
                        grad_norms_[name] = torch.nn.utils.clip_grad_norm_(
                            [param],
                            max_norm = config.TRAIN.CLIP_GRAD,
                            error_if_nonfinite=True
                        ).detach().cpu().item()
                    spectral_grad_norm = sum(grad_norms_.values())
                # TODO Clip in chroma net
            else:
                spectral_grad_norm = -1.0
                chroma_grad_norm   = -1.0

            # Optimize
            optimizer.step()
            lr_scheduler.step()

            # Update meters
            elapsed_time = time.time() - tic
            batch_time_meter.update(elapsed_time)
            total_loss_meter.update(loss["total"].item())
            spectral_loss_meter.update(loss["spectral"].item())
            chroma_loss_meter.update(loss["chroma"].item())
            spectral_grad_norm_meter.update(spectral_grad_norm)
            # chroma_grad_norm_meter.update(chroma_grad_norm.cpu().numpy())

            # Checkpoint models
            if current_iter % config.TRAIN.CHECKPOINT_EVERY == 0:
                savestate = {
                    "spectral": spectral_net.state_dict(),
                    "chroma": chroma_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "iteration": current_iter,
                    "psnr": -1
                }
                save_checkpoint(savestate, config, logger)

            # Log
            if current_iter % config.LOGGING.LOG_EVERY == 0:
                lr = lr_scheduler.get_last_lr()[0]
                wd = optimizer.param_groups[0]["weight_decay"]
                memory_used = get_alloc_memory(config)
                eta = batch_time_meter.mean * (max_iters - current_iter)

                logger.info(
                    f"Train: [{current_iter:>6}/{max_iters:>6}]\t"
                    f"eta: {datetime.timedelta(seconds=int(eta))}\t"
                    f"lr: {lr:.6f}\t wd {wd:.4f}\t"
                    f"time: {batch_time_meter.mean:.4f} ± {batch_time_meter.std:.4f}\t"
                    f"spectral loss: {spectral_loss_meter.mean:.4f} ± {spectral_loss_meter.std:.4f}\t"
                    f"chroma loss: {chroma_loss_meter.mean:.4f} ± {chroma_loss_meter.std:.4f}\t"
                    f"total loss: {total_loss_meter.mean:.4f} ± {total_loss_meter.std:.4f}\t"
                    f"spectral grad norm: {spectral_grad_norm_meter.mean:.4f} ± {spectral_grad_norm_meter.std:.4f}\t"
                    # f"chroma grad norm: {chroma_grad_norm_meter.mean:.4f} ± {chroma_grad_norm_meter.std:.4f}\t"
                    f"memory: {memory_used:.0f}MB"
                )

                # Reset meters
                batch_time_meter.reset()
                total_loss_meter.reset()
                spectral_loss_meter.reset()
                chroma_loss_meter.reset()
                spectral_grad_norm_meter.reset()
                # chroma_grad_norm_meter.reset()

                # Log individual gradient norms
                for k, v in grad_norms_.items():
                    logger.info(f"\tgrad_norm({k:<80}) = {v:.4f}")

                # Log relative updates
                for k, val in spectral_net.named_parameters():
                    if k in param_values_:
                        x0 = torch.linalg.norm(val.detach().cpu() - param_values_[k])
                        x1 = x0 / (torch.linalg.norm(param_values_[k]) + 1e-6)
                        logger.info(f"\tupdate_norm({k:<80}) = {x0.item():.4f} / {x1.item():.4f}")

        if current_iter >= max_iters:
            break


def get_alloc_memory(config):
    if config.TRAIN.DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / (2 ** 20)
    elif config.TRAIN.DEVICE == "mps":
        return torch.mps.current_allocated_memory() / (2 ** 20)
    else:
        return -1.0



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