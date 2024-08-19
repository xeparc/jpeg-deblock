"""This scrips calculates the mean and std of DCT coefficients per 8x8 block
for images inside a single or multiple directories."""

import argparse
import json
import os

import numpy as np
import torchvision
from tqdm import tqdm

import context
from jpegutils import JPEGImageData
from utils import RunningMeanStd


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=str)
    parser.add_argument("--output", "-o", action="store", type=str,
                        default="dct-stats.json")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args()

    # Create output file
    outpath = args.output
    if not outpath.lower().endswith(".json"):
        outpath = outpath + ".json"
    if os.path.exists(outpath):
        overwrite = 'x'
        while not (overwrite in 'yn'):
            overwrite = input("Output file exists! Overwrite? [y/n]: ").lower()
        if overwrite == 'n':
            exit()
        else:
            open(outpath, mode="wt").close()

    # Walk down the directory tree and collect image paths
    image_paths = []
    for _dir in args.directories:
        if not os.path.isdir(_dir):
            continue
        for rootdir, _, filenames in os.walk(_dir):
            for file in filenames:
                fullpath = os.path.join(rootdir, file)
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    image_paths.append(fullpath)

    iterable = tqdm(image_paths) if not args.verbose else image_paths
    estimator_Y = RunningMeanStd()
    estimator_C = RunningMeanStd()

    # Calculate DCT coefficients mean and std
    for impath in iterable:
        rgb = torchvision.io.read_image(impath)
        dat = JPEGImageData(rgb.permute(1,2,0).numpy(), quality=100)
        dctY  = dat.dctY.numpy()
        dctCb = dat.dctCb.numpy()
        dctCr = dat.dctCr.numpy()
        dctC = np.stack([dctCb, dctCr], axis=0)
        assert dctC.shape[0] == 2
        assert dctC.shape[3] == 8
        assert dctC.shape[4] == 8

        for block in dctY.reshape(-1, 8, 8):
            estimator_Y.update(block)

        for block in dctC.reshape(-1, 8, 8):
            estimator_C.update(block)

        if args.verbose:
            print(impath)

    stats = {
        "dct_Y_mean": estimator_Y.mean,
        "dct_C_mean": estimator_C.mean,
        "dct_Y_std":  estimator_Y.std,
        "dct_C_std":  estimator_C.std
    }

    with open(outpath, mode="wt") as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=2)
