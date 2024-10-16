import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image

from builders import build_model
from config import get_config
from dataset import EvaluationDataset
from inference import enhance_color, enhance_grayscale
from metrics import calculate_psnr, calculate_psnrb, calculate_ssim
from models.flare import FlareLuma


LIVE1_COMPRESSED_COLOR = "data/testsets/live1/color/"
LIVE1_COMPRESSED_GRAY = "data/testsets/live1/gray/"
LIVE1_ORIGINALS  = "data/testsets/live1/refimgs/"
LIVE1_ORIGINALS_GRAY = "data/testsets/live1/refimgs-gray/"

CLASSIC5_COMPRESSED = "data/testsets/classic5/gray/"
CLASSIC5_ORIGINALS = "data/testsets/classic5/refimgs/"

BSDS500_COMPRESSED_COLOR = "data/testsets/bsds500/color/"
BSDS500_COMPRESSED_GRAY = "data/testsets/bsds500/gray/"
BSDS500_ORIGINALS = "data/testsets/bsds500/refimgs/"



def evaluate_live1(model, color=True, enhance=True, savedir='', verbose=True) -> dict:
    Qs = (10, 20, 30, 40, 50, 60, 70, 80, 90)
    # Qs = (10, 20, 30, 40)
    result = {}
    for q in Qs:
        if savedir:
            _savedir = os.path.join(savedir, f"qf_{q}")
            os.makedirs(_savedir, exist_ok=True)
        else:
            _savedir = ''
        cdir = LIVE1_COMPRESSED_COLOR if color else LIVE1_COMPRESSED_GRAY
        compressed_dir = os.path.join(cdir, f"qf_{q}")
        originals_dir = LIVE1_ORIGINALS if color else LIVE1_ORIGINALS_GRAY
        metrics = evaluate_dir(compressed_dir, originals_dir, model,
                               color=color, enhance=enhance, verbose=verbose,
                               savedir=_savedir)
        result[q] = metrics
    return result


def evaluate_classic5(model, enhance=True, savedir='', verbose=True) -> dict:
    Qs = (10, 20, 30, 40, 50, 60, 70, 80, 90)
    # Qs = (10, 20, 30, 40)
    result = {}
    for q in Qs:
        if savedir:
            _savedir = os.path.join(savedir, f"qf_{q}")
            os.makedirs(_savedir, exist_ok=True)
        else:
            _savedir = ''
        compressed_dir = os.path.join(CLASSIC5_COMPRESSED, f"qf_{q}")
        metrics = evaluate_dir(compressed_dir, CLASSIC5_ORIGINALS, model,
                               color=False, enhance=enhance, verbose=verbose,
                               savedir=_savedir)
        result[q] = metrics
    return result


def evaluate_bsds500(model, color=True, enhance=True, savedir='', verbose=True) -> dict:
    Qs = (10, 20, 30, 40, 50, 60, 70, 80, 90)
    # Qs = (10, 20, 30, 40)
    result = {}
    for q in Qs:
        if savedir:
            _savedir = os.path.join(savedir, f"qf_{q}")
            os.makedirs(_savedir, exist_ok=True)
        else:
            _savedir = ''
        cdir = BSDS500_COMPRESSED_COLOR if color else BSDS500_COMPRESSED_GRAY
        compressed_dir = os.path.join(cdir, f"qf_{q}")
        metrics = evaluate_dir(compressed_dir, BSDS500_ORIGINALS, model,
                               color=color, enhance=enhance, verbose=verbose,
                               savedir=_savedir)
        result[q] = metrics
    return result



def evaluate_dir(compressed_dir, originals_dir, model, color=True, savedir='',
                 enhance=True, verbose=False) -> dict:

    if savedir:
        os.makedirs(savedir, exist_ok=True)

    if verbose:
        print(f"\nEvaluating images in \"{compressed_dir}\", color={color}, enhance={enhance}\n",
              flush=True)

    dataset = EvaluationDataset(compressed_dir, originals_dir)
    psnrs  = []
    psnrbs = []
    ssims  = []
    names  = []
    for i in range(len(dataset)):
        x_path, y_path = dataset[i]
        if color:
            if enhance:
                x = enhance_color(x_path, model, output_dtype=torch.float32)
            else:
                x = Image.open(x_path)
                x = torchvision.transforms.functional.to_tensor(x)
            target = Image.open(y_path)
            target = torchvision.transforms.functional.to_tensor(target)
        else:
            if enhance:
                x = enhance_grayscale(x_path, model, output_dtype=torch.float32)
            else:
                x = Image.open(x_path).convert("L")
                x = torchvision.transforms.functional.to_tensor(x)
            target = Image.open(y_path).convert("L")
            target = torchvision.transforms.functional.to_tensor(target)
        psnr = calculate_psnr(x, target, channel_dim="first")
        psnrb = calculate_psnrb(x, target, channel_dim="first")
        ssim = calculate_ssim(x, target, channel_dim="first")
        psnrs.append(psnr)
        psnrbs.append(psnrb)
        ssims.append(ssim)
        names.append(os.path.basename(x_path))

        if verbose:
            print(f"{os.path.basename(x_path):<40}: {psnr:.3f} / {psnrb:.3f} / {ssim:.3f}",
                  flush=True)

        if savedir:
            name, _ = os.path.splitext(os.path.basename(x_path))
            img = torchvision.transforms.ConvertImageDtype(torch.uint8)(x)
            savepath = os.path.join(savedir, f"{name}.png")
            torchvision.io.write_png(img, savepath)

    return {
        "psnr": np.mean(psnrs),
        "psnrb": np.mean(psnrbs),
        "ssim": np.mean(ssims),
        # "psnr_detail": [[n, p] for n, p in zip(names, psnrs)]
    }


def evaluate_model(logpath, save_results='', savedir='', verbose=True):

    config = get_config(os.path.join(logpath, "config.yaml"))
    model = build_model(config)

    state = torch.load(
        os.path.join(logpath, "model.pth"),
        map_location="cpu",
        weights_only=True
    )

    res = model.load_state_dict(state)
    print(res, flush=True)

    if isinstance(model, FlareLuma):
        classic5_results = evaluate_classic5(
            model, verbose=verbose, savedir=os.path.join(savedir, "Classic5")
        )
        live1_gray_results = evaluate_live1(
            model, color=False, verbose=verbose,
            savedir=os.path.join(savedir, "Live1_gray")
        )
        bsds500_gray_results = evaluate_bsds500(
            model, color=False, verbose=verbose,
            savedir=os.path.join(savedir, "BSDS500_gray")
        )
        results = {
            "Live1.gray":       live1_gray_results,
            "Classic5.gray":    classic5_results,
            "BSDS500.gray":     bsds500_gray_results
        }
    # Evaluate color only if model supports it
    else:
        live1_color_results = evaluate_live1(
            model, color=True, verbose=verbose,
            savedir=os.path.join(savedir, "Live1_color")
        )
        bsds500_color_results = evaluate_bsds500(
            model, color=True, verbose=verbose,
            savedir=os.path.join(savedir, "BSDS500_color")
        )
        results = {
            "Live1.color":      live1_color_results,
            "BSDS500.color":    bsds500_color_results,
        }

    if save_results:
        os.makedirs(os.path.dirname(save_results), exist_ok=True)
        with open(save_results, mode="wt") as f:
            json.dump(results, f, indent=2, sort_keys=True)

    return save_results


if __name__ == "__main__":

    model_paths = {
        "Flare.Luma.T": "logs/FlareLuma.Tiny-final/",
        "Flare.Luma.S": "logs/FlareLuma.Small-final/",
        "Flare.Luma.M": "logs/FlareLuma.Medium-final/",
        "Flare.Luma.L": "logs/FlareLuma.Large-final/",

        "Flare.Chroma.T": "logs/Flare.Chroma.Tiny-final/",
        "Flare.Chroma.S": "logs/Flare.Chroma.Small-final/",
        "Flare.Chroma.M": "logs/Flare.Chroma.Medium-final/",
        "Flare.Chroma.L": "logs/Flare.Chroma.Large-final/",

        "Flare.T": "logs/Flare.Tiny-final/",
        "Flare.S": "logs/Flare.Small-final/",
        "Flare.M": "logs/Flare.Medium-final/",
        "Flare.L": "logs/Flare.Large-final/",

        "Flare.Luma.L400": "logs/Flare.Luma.Large-final-400k/",
        "Flare.Chroma.L400": "logs/Flare.Chroma.Large-final-400k/"
    }

    results_dir = os.path.join("data/results/")
    os.makedirs(results_dir, exist_ok=True)

    for key, path in model_paths.items():
        print(f"\n\nEvaluating \"{key}\"...\n", flush=True)
        savepath = os.path.join(results_dir, key + ".json")
        imagedir = f"data/eval/{key}/"
        results = evaluate_model(path, save_results=savepath, savedir=imagedir, verbose=True)

    # res = evaluate_bsds500(None, enhance=False, verbose=False, color=False)
    # print(res)