import argparse
import sys
import os

from turbojpeg import TurboJPEG


def encode_and_save_jpegs(input_dir, output_dir, quality=(80, 60, 40, 30, 20, 10)):
    """
    Re-encodes every JPEG image in `input_dir` (recursively) with quality Q and
    saves it in `output_dir`/Q/. The directory structure below `input_dir` is
    mirrored under `output_dir`/Q/

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
        savedir = os.path.join(output_dir, str(q))
        os.makedirs(savedir, exist_ok=True)

        for rootdir, _, filenames in os.walk(input_dir):
            prefix = os.path.commonpath([rootdir, input_dir])
            subpath = os.path.join(savedir, rootdir[len(prefix)+1:])
            os.makedirs(subpath, exist_ok=True)
            for file in filenames:
                if not file.endswith(".jpg") and not file.endswith(".jpeg"):
                    continue
                with open(os.path.join(rootdir, file), mode="rb") as f:
                    data = f.read()
                    img = turbo.decode(data)
                # Save
                encoded = turbo.encode(img, quality=q)
                savepath = os.path.join(subpath, file)
                with open(savepath, mode="wb") as f:
                    f.write(encoded)


if __name__ == "__main__":
    pass