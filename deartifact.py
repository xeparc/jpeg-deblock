"""Command line utility to de-artifact a single JPEG image"""
import argparse
import os

import torch
import torchvision

from builders import build_model
from config import get_config
from inference import enhance_color, enhance_grayscale


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str, default='')
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--gray", "-g", action="store_true", default=False)
    args = parser.parse_args()

    config = get_config(os.path.join(args.model, "config.yaml"))
    model = build_model(config)
    state = torch.load(os.path.join(args.model, "model.pth"), map_location="cpu")
    res = model.load_state_dict(state)
    print(res)

    if args.gray:
        enhanced = enhance_grayscale(args.input, model, output_dtype=torch.uint8)
    else:
        enhanced = enhance_color(args.input, model, torch.uint8)

    if not args.output:
        name, _ = os.path.splitext(os.path.basename(args.input))
        savepath = name + ".png"
    else:
        savepath = args.output
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

    torchvision.io.write_png(enhanced, savepath)
