import os
import pickle
import torch
import torch.optim.adam
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils

from models import SRCNN


TRAIN_PATH = "BSDS500/BSDS500/data/images/train/"
VAL_PATH = "BSDS500/BSDS500/data/images/val/"
TEST_PATH = "BSDS500/BSDS500/data/images/test/"

DEBUG = True

# TODO
# Add image augmentation - rotation, scaling
# Add regularization



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
class DatasetRescale(torch.utils.data.Dataset):

    def __init__(self, path, scale=2):
        super().__init__()
        assert os.path.exists(path)
        self.img_dir = path
        self.scale = scale
        bilinear = torchvision.transforms.InterpolationMode.BILINEAR
        self.tofloat = torchvision.transforms.ConvertImageDtype(torch.float32)

        # Load images
        self.inputs = []
        self.targets = [] 
        for item in os.scandir(self.img_dir):
            if item.name.endswith(".jpg"):
                img = torchvision.io.read_image(item.path)
                c, h, w = img.shape
                if h > w:
                    img = img.transpose(1,2)
                    h, w = w, h
                assert img.shape[0] < img.shape[1]

                downscale = torchvision.transforms.Resize(
                    size=(h // self.scale, w // self.scale),
                    interpolation=bilinear)
                upscale = torchvision.transforms.Resize(
                    size=(h,w), interpolation=bilinear)
                x = upscale(downscale(img))
                y = img
                self.inputs.append(x)
                self.targets.append(y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image, target = self.inputs[idx], self.targets[idx]
        # Convert to float
        return self.tofloat(image), self.tofloat(target)


class DatasetRescalePatches(torch.utils.data.Dataset):

    def __init__(self, path, scale=2, patchsize=40):
        super().__init__()
        assert os.path.exists(path)
        self.imgdir = path
        self.scale = scale
        self.patchsize = patchsize
        bilinear = torchvision.transforms.InterpolationMode.BILINEAR
        self.tofloat = torchvision.transforms.ConvertImageDtype(torch.float32)

        # Load images
        self.inputs = []
        self.targets = []
        for item in os.scandir(self.imgdir):
            if item.name.endswith(".jpg"):
                img = torchvision.io.read_image(item.path)
                c, h, w = img.shape
                if h > w:
                    img = img.transpose(1,2)
                    h, w = w, h
                assert img.shape[0] < img.shape[1]

                downscale = torchvision.transforms.Resize(
                    size=(h // self.scale, w // self.scale),
                    interpolation=bilinear)
                upscale = torchvision.transforms.Resize(
                    size=(h,w), interpolation=bilinear)
                x = upscale(downscale(img))
                # Cut into patches
                x_patches = cutp(x[:, :-1, :-1], patchsize, patchsize)
                y_patches = cutp(img[:, :-1, :-1], patchsize, patchsize)
                self.inputs.extend(x_patches)
                self.targets.extend(y_patches)


    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        image, target = self.inputs[idx], self.targets[idx]
        # Convert to float
        return self.tofloat(image), self.tofloat(target)


def train_loop(dataloader, model, loss_fn, optimizer, device):

    size = len(dataloader.dataset)

    train_history = []

    model.train()
    for batch_num, (X, y) in enumerate(dataloader):
        pred = model(X.to(device=device))
        loss = loss_fn(pred, y.to(device=device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_num % 20) == 0:
            loss, current = loss.item(), batch_num * 128
            psnr = 10 * np.log10(1 / loss)
            print(f"loss: {loss:>4f}, psnr: {psnr:>4f}, [{current:>5d}/{size:>5d}]")
            train_history.append(loss)

    return train_history


if __name__ == "__main__":

    batch_size = 32
    learning_rate = 4e-4
    version = 0.6
    epochs = 1000
    device = "mps"
    model_savepath = f"models/SRCNN-v{version}.pt"

    train_data = DatasetRescalePatches(TRAIN_PATH, scale=2)
    val_data = DatasetRescalePatches(VAL_PATH, scale=2)
    test_data = DatasetRescalePatches(TEST_PATH, scale=2)
    print(len(train_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

    device = torch.device(device)
    loss_fn = nn.MSELoss()
    net = SRCNN().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    train_history = []
    for ep in range(epochs):
        print(f"Epochs {ep + 1}\n-----------------------------")
        hist = train_loop(train_loader, net, loss_fn, optimizer, device)
        train_history.extend(hist)

        # Save model
        torch.save(net.state_dict(), model_savepath)

        # Save train history
        with open(f"models/history-v{version}.pickle", mode="wb") as f:
            pickle.dump(train_history, f)

    print("Done!")

