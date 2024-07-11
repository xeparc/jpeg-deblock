import os
import re
import torch
import torch.nn as nn
import torchvision
from turbojpeg import TurboJPEG


IMAGES_PATH = "data/BSDS500/BSDS500/data/images/"
OUTPUT_PATH = "data/"
QUALITY = [10, 20, 30, 40, 50, 60, 80]
DEBUG = 0


def cutp(image, height, width):
    """Cuts `image` into nonoverlapping patches."""
    C, H, W = image.shape
    pad_w = W % width
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_h = H % height
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded_image = nn.functional.pad(
        image, (pad_left, pad_right, pad_top, pad_bottom))

    # (C, H, W)
    patches = padded_image.unfold(dimension=1, size=height, step=height)
    # (C, nH, W, height)
    patches = patches.unfold(dimension=2, size=width, step=width)
    # (C, nH, nW, height, width)

    # Test if they are the same
    if DEBUG:
        patches_orig = patches.permute(0, 1, 3, 2, 4).contiguous()
        assert torch.all(patches_orig.reshape(image.shape) == image)

    return patches.reshape(C, -1, height, width).transpose(0, 1)


# A custom Dataset class must implement three functions:
#       __init__, __len__, and __getitem__.
class DatasetDeblockPatches(torch.utils.data.Dataset):

    def __init__(self, compressed_dir, originals_dir, subpatch_size=20, device="cpu"):
        super().__init__()
        assert os.path.exists(compressed_dir)
        assert os.path.exists(originals_dir)

        self.compressed_dir = os.path.normpath(compressed_dir)
        self.originals_dir  = os.path.normpath(originals_dir)
        self.subpatch_size = subpatch_size
        self.tofloat = torchvision.transforms.ConvertImageDtype(torch.float32)
        self.device = torch.device(device)

        # Load images
        self.inputs = []
        self.targets = []
        for item in os.scandir(self.compressed_dir):
            if item.name.lower().endswith(".jpg") or item.name.lower().endswith(".jpeg"):
                # Find the pairing target image in `originals_dir` for `item`
                target_path = os.path.join(self.originals_dir, item.name)
                # Load images from disk
                trainimg = torchvision.io.read_image(item.path)
                targetimg = torchvision.io.read_image(target_path)
                c, h, w = trainimg.shape
                # Transpose if image is portrait
                if h > w:
                    trainimg = trainimg.transpose(1,2)
                    targetimg = targetimg.transpose(1,2)
                    h, w = w, h
                assert trainimg.shape[0] < trainimg.shape[1]

                # Cut into patches
                sps = self.subpatch_size
                x_patches = cutp(trainimg[:, :-1, :-1], sps, sps)
                y_patches = cutp(targetimg[:, :-1, :-1], sps, sps)
                self.inputs.extend(x_patches)
                self.targets.extend(y_patches)


    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        image, target = self.inputs[idx], self.targets[idx]
        # Convert to float and move to `self.device`
        x = self.tofloat(image).to(self.device)
        y =  self.tofloat(target).to(self.device)
        return x, y


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

