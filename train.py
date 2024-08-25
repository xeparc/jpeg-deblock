import argparse
import datetime
import os
import time
from typing import List, Callable
import yaml

from yacs.config import CfgNode

from config import default_config, get_config
from builders import *
from utils import RunningStats, save_checkpoint, load_checkpoint
from jpegutils import SUBSAMPLE_FACTORS


def main(cfg):

    device = torch.device(cfg.TRAIN.DEVICE)

    train_loader    = build_dataloader(cfg, "train")
    val_loader      = build_dataloader(cfg, "val")
    test_loader     = build_dataloader(cfg, "test")

    spectre         = build_spectral_net(cfg).to(device)
    chroma          = build_chroma_net(cfg).to(device)
    criterion       = build_criterion(cfg)

    optim           = build_optimizer(cfg, spectre.parameters(), chroma.parameters())
    lr_scheduler    = build_lr_scheduler(cfg, optim)
    logger          = build_logger(cfg)

    if cfg.TRAIN.CHECKPOINT_EVERY > 0:
        os.makedirs(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, cfg.MODEL.NAME),
                    exist_ok=True)

    train_loop(cfg, spectre, chroma, criterion, train_loader, optim,
               lr_scheduler, logger)



def predict_train(spectral_net, chroma_net, datapoint, subsample: str):
    input_yy_dct =  datapoint["lq_dct_y"]
    input_cb_dct =  datapoint["lq_dct_cb"]
    input_cr_dct =  datapoint["lq_dct_cr"]
    qt_luma =       datapoint["qt_y"]
    qt_chroma =     datapoint["qt_c"]

    out_yy = spectral_net(input_yy_dct, qt_luma, chroma=False)
    out_cb = spectral_net(input_cb_dct, qt_chroma, chroma=True)
    out_cr = spectral_net(input_cr_dct, qt_chroma, chroma=True)

    scale = SUBSAMPLE_FACTORS[subsample]

    # Upsample chroma components
    upsampled_cb = torch.nn.functional.interpolate(
        out_cb, scale_factor=scale, mode="nearest"
    )
    upsampled_cr = torch.nn.functional.interpolate(
        out_cr, scale_factor=scale, mode="nearest"
    )

    # Concatenate Y, Cb+, Cr+ planes
    ycc = torch.concatenate([out_yy, upsampled_cb, upsampled_cr], dim=1)
    enhanced = chroma_net(ycc)
    assert ycc.shape[1] == 3

    return {"Y": out_yy, "Cb": out_cb, "Cr": out_cr, "final": enhanced}


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

            # Clip gradients
            if clip_grad:
                spectral_grad_norm = torch.nn.utils.clip_grad_norm_(
                    spectral_net.parameters(),
                    max_norm = config.TRAIN.CLIP_GRAD,
                    error_if_nonfinite=True
                )
                chroma_grad_norm = torch.nn.utils.clip_grad_norm_(
                    chroma_net.parameters(),
                    max_norm = config.TRAIN.CLIP_GRAD,
                    error_if_nonfinite=True
                )
            else:
                spectral_grad_norm = -1.0
                chroma_grad_norm   = -1.0

            # Optimize
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Update meters
            elapsed_time = time.time() - tic
            batch_time_meter.update(elapsed_time)
            total_loss_meter.update(loss["total"].item())
            spectral_loss_meter.update(loss["spectral"].item())
            chroma_loss_meter.update(loss["chroma"].item())
            spectral_grad_norm_meter.update(spectral_grad_norm.cpu().numpy())
            chroma_grad_norm_meter.update(chroma_grad_norm.cpu().numpy())

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
                    f"chroma grad norm: {chroma_grad_norm_meter.mean:.4f} ± {chroma_grad_norm_meter.std:.4f}\t"
                    f"memory: {memory_used:.0f}MB"
                )

                # Reset meters
                batch_time_meter.reset()
                total_loss_meter.reset()
                spectral_loss_meter.reset()
                chroma_loss_meter.reset()
                spectral_grad_norm_meter.reset()
                chroma_grad_norm_meter.reset()

        if current_iter >= max_iters:
            break


def validate():
    pass


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