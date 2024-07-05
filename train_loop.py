import argparse
import os
import pickle
import re
import torch
import torch.optim.adam
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils

from models import ARCNN
from dataset import DatasetDeblockPatches


ORIGINALS_TRAIN_PATH = "data/100/train/"
ORIGINALS_VAL_PATH   = "data/100/val/"
ORIGINALS_TEST_PATH  = "data/100/test/"

TRAIN_PATH = "data/80/train/"
VAL_PATH   = "data/80/val/"
TEST_PATH  = "data/80/test/"

DEBUG = True

# TODO
# Add image augmentation - rotation, scaling
# Add regularization


def validation_loop(dataloader, model, loss_fn, device):
    pass


def test_loop(dataloader, model, loss_fn, device):
    pass


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

    batch_size = 64
    learning_rate = 4e-3
    version = 0.3
    quality = 80
    epochs = 100
    
    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


    modelname = f"ARCNN-quality{quality}-v{version}"
    model_savepath = f"models/{modelname}.pt"
    optimizer_savepath = f"models/{modelname}-optimizer.pt"

    train_data = DatasetDeblockPatches(TRAIN_PATH)
    val_data = DatasetDeblockPatches(VAL_PATH)
    test_data = DatasetDeblockPatches(TEST_PATH)
    print(len(train_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

    device = torch.device(device)
    loss_fn = nn.MSELoss()
    net = ARCNN().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    train_history = []
    for ep in range(epochs):
        print(f"Epochs {ep + 1}\n-----------------------------")
        hist = train_loop(train_loader, net, loss_fn, optimizer, device)
        train_history.extend(hist)

        # Save model
        torch.save(net.state_dict(), model_savepath)

        # Save optimizer
        torch.save(optimizer, optimizer_savepath)

        # Save train history
        with open(f"models/{modelname}-history.pickle", mode="wb") as f:
            pickle.dump(train_history, f)

    print("Done!")
