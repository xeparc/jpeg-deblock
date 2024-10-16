import argparse
import os

import numpy as np
import turbojpeg
from turbojpeg import TurboJPEG
from PIL import Image

SUBSAMPLE_FACTORS = {
    444: "4:4:4",
    422: "4:2:2",
    420: "4:2:0",
}


def compress_jpegs(input_dir, output_dir, quality=(80, 60, 40, 30, 20, 10),
                   subsample=444, grayscale=False):
    """
    Re-encodes every image in `input_dir` (recursively) with quality Q and
    saves it in `output_dir`/Q/.

    Parameters:
    -----------
        input_dir: str
            The source directory from which JPEG images are read.
        output_dir: str
            The output directory in which re-encoded JPEG images are saved.
            If it does not exists, it's created.
        quality: iterable
            Quality settings passed to TurboJPEG encoder. For each Q in
            `quality`, a directory is created in `output_dir`. The directory
            structure if `input_dir`/ is mirrored in `output_dir`/Q/.

    Returns:
    --------
        None
    """

    # Create `output_dir` if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    turbo = TurboJPEG()

    # Encode loop
    for q in quality:
        savedir = os.path.join(output_dir, f"qf_{q}")
        os.makedirs(savedir, exist_ok=True)

        for rootdir, _, filenames in os.walk(input_dir):
            prefix = os.path.commonpath([rootdir, input_dir])
            subpath = os.path.join(savedir, rootdir[len(prefix)+1:])
            os.makedirs(subpath, exist_ok=True)
            for file in filenames:
                name, _ = os.path.splitext(file)
                # Open
                path = os.path.join(rootdir, file)
                try:
                    img = Image.open(path)
                except:
                    continue
                if grayscale:
                    img = img.convert("L")
                # Save
                savepath = os.path.join(savedir, name + ".jpg")
                img.save(savepath, format="jpeg",
                         quality=q,
                         subsampling=SUBSAMPLE_FACTORS[subsample]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-i", type=str)
    parser.add_argument("--destination", "-d", type=str)
    parser.add_argument("--subsample", "-c", type=int, default=444)
    parser.add_argument("--gray", "-g", action="store_true", default=False)
    args = parser.parse_args()

    Qs = [10,20,30,40,50,60,70,80,90]
    compress_jpegs(args.source, args.destination, Qs, args.subsample, args.gray)