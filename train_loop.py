import argparse
import datetime
import json
import os
import pickle
import re
import torch
import torch.optim.adam
import torch.optim.adam
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils

import models
from dataset import DatasetDeblockPatches


ORIGINALS_TRAIN_PATH    = "data/100/train/"
ORIGINALS_VAL_PATH      = "data/100/val/"
ORIGINALS_TEST_PATH     = "data/100/test/"
TRAIN_PATH              = "data/80/train/"
VAL_PATH                = "data/80/val/"
TEST_PATH               = "data/80/test/"
DNN_SAVEPATH            = "models/"

DEBUG = True


# TODO
# 0. Overfit a small batch!
# 1. Log updates magnitude
# 2. Log convolutional filters
# 3. Use learning rate decay
# 4. Use another loss function, PSNR-B
# 5. Use deeper DNN arhchitecture
# 6. Add ReduceLROnPlateau


def test_loop(net, dataloader):
    loss_fn = nn.MSELoss()
    mse_list, snr_list = [], []
    with torch.no_grad():
        for X, y in dataloader:
            pred = net(X)
            mse = loss_fn(pred, y)
            mse_list.append(mse.item())
            snr_list.append(10 * np.log10(1 / mse.item()))
    return {"mse": np.mean(mse_list), "psnr": np.mean(snr_list)}


def epoch_loop(net, dataloader, loss_fn, optimizer, print_every=20):

    size = len(dataloader.dataset)
    updates = 0

    mse_list, snr_list = [], []

    net.train()

    for batch_num, (X, y) in enumerate(dataloader):
        pred = net(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        updates += 1

        # Keep stats
        mse_list.append(loss.item())
        snr_list.append(10 * np.log10(1 / loss.item()))

    # Print
    mean_loss = np.mean(mse_list)
    mean_psnr = np.mean(snr_list)
    print(f"avg loss: {mean_loss:>4f}, psnr: {mean_psnr:>4f}")
    return {"mse": mean_loss, "psnr": mean_psnr, "updates": updates}


def train_loop(net, optimizer, train_loader, val_loader, n_epochs, output_dir,
               checkpoint_every=1):

    loss_fn = nn.MSELoss()

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir)
    net_name = net.__class__.__name__

    train_metrics = {"epoch": [], "updates": [], "mse": [], "psnr": []}
    validation_metrics = {"epoch": [], "mse": [], "psnr": []}

    for ep in range(n_epochs):
        print(f"Epochs {ep + 1}\n-----------------------------")

        epoch_stats = epoch_loop(net, train_loader, loss_fn, optimizer)

        # Save train metrics
        train_metrics["epoch"].append(ep)
        train_metrics["mse"].append(epoch_stats["mse"])
        train_metrics["psnr"].append(epoch_stats["psnr"])
        train_metrics["updates"].append(epoch_stats["updates"])

        with open(os.path.join(output_dir, "train-stats.json"), mode="wt") as f:
            json.dump(train_metrics, f)

        # Checkpoint
        if (ep % checkpoint_every) == 0:
            # Save optimizer
            torch.save(optimizer, os.path.join(checkpoint_dir, f"optimizer-{ep}.pt"))
            # Save net
            torch.save(net, os.path.join(checkpoint_dir, f"{net_name}-{ep}.pt"))

        # Evaluate on validation set
        vmetrics = test_loop(net, val_loader)
        validation_metrics["epoch"].append(ep)
        validation_metrics["mse"].append(vmetrics["mse"])
        validation_metrics["psnr"].append(vmetrics["psnr"])

        # Save validation metrics
        with open(os.path.join(output_dir, "val-stats.json"), mode="wt") as f:
            json.dump(validation_metrics, f)

    # Save trained model
    torch.save(net.state_dict(), os.path.join(output_dir, net_name + ".pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="train-jpeg-deblock",
        description="Train JPEG Deblock DNN."
    )
    parser.add_argument("--compressed_dir", action="store", type=str, required=True,
                        help="Path to directory with compressed images")
    parser.add_argument("--originals_dir", action="store", type=str, required=True,
                        help="Path to directory with original images")
    parser.add_argument("--subpatch_size", action="store", type=int, default=20,
                        help="Size of extracted subpatches")
    parser.add_argument("--model_name", action="store", type=str, required=True,
                        help="Name of DNN model to use")
    parser.add_argument("--batch_size", action="store", type=int, default=64,
                        help="Input batch size")
    parser.add_argument("--lr", action="store", type=float, default=4e-3,
                        help="Learning rate")
    parser.add_argument("--device", action="store", type=str, default="cpu",
                        help="Device to be used for training")
    parser.add_argument("--epochs", "-e", action="store", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--n_updates", action="store", type=int, default=-1,
                        help="Number of updates for which the model is trained.")
    parser.add_argument("--checkpoint_every", action="store", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--suffix", action="store", type=str, default="",
                        help="Suffix string added to output directory")
    parser.add_argument("--optimizer", action="store", type=str, default="sgd",
                        help="Optmiization algorithm to use")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--checkerboard", action="store_true", default=False)
    parser.add_argument("--load_params", action="store", type=str, default='',
                        help="Load model from checkpoint")
    parser.add_argument("--load_optimizer", action="store", type=str, default='',
                        help="Load optimizer from checkpoint")

    args = parser.parse_args()

    # Check that `compressed_dir` and `originals_dir` contain train/ test/ val/
    # subfolders
    for subpath in ("train", "test", "val"):
        assert os.path.exists(os.path.join(args.compressed_dir, subpath))
        assert os.path.exists(os.path.join(args.originals_dir, subpath))

    # Initialize neural net
    device = torch.device(args.device)
    model_name = args.model_name
    channels = 4 if args.checkerboard else 3
    net = getattr(models, model_name)(channels).to(device)
    # ... and maybe load checkpoint
    if args.load_params:
        net = torch.load(args.load_params)
    net.train()

    # Load optimizer from checkpoint
    if args.load_optimizer:
        optimizer = torch.load(args.load_optimizer)
    else:
    # ... or initialize optimizer
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    optimizer.param_groups[0]["lr"] = 1e-5
    optimizer.zero_grad()

    # Create output directory to save DNN model & it's checkpoints
    output_dir = os.path.join("models", model_name + args.suffix)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize data loaders
    train_data = DatasetDeblockPatches(
        os.path.join(args.compressed_dir, "train"),
        os.path.join(args.originals_dir, "train"),
        subpatch_size=args.subpatch_size,
        normalize=args.checkerboard,
        checkerboard=args.checkerboard,
        device=device
    )
    val_data = DatasetDeblockPatches(
        os.path.join(args.compressed_dir, "val"),
        os.path.join(args.originals_dir, "val"),
        subpatch_size=args.subpatch_size,
        normalize=args.checkerboard,
        checkerboard=args.checkerboard,
        device=device
    )
    test_data = DatasetDeblockPatches(
        os.path.join(args.compressed_dir, "test"),
        os.path.join(args.originals_dir, "test"),
        subpatch_size=args.subpatch_size,
        normalize=args.checkerboard,
        checkerboard=args.checkerboard,
        device=device
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, args.batch_size, shuffle=True
    )
    val_loader   = torch.utils.data.DataLoader(
        val_data, args.batch_size, shuffle=True
    )
    test_loader  = torch.utils.data.DataLoader(
        test_data, args.batch_size, shuffle=True
    )

    # Print train session info to file on disk
    traininfo_path = os.path.join(output_dir, "traininfo.txt")
    fmode = "a" if os.path.exists(traininfo_path) else "w"
    with open(traininfo_path, mode=fmode) as f:
        today = datetime.datetime.today()
        print("Train session started at", today.isoformat(timespec="seconds"), file=f)
        print("=" * 80, file=f)
        print("Output directory:", output_dir, file=f)
        print("Compressed images directory:", args.compressed_dir, file=f)
        print("Original images directory:", args.originals_dir, file=f)
        print("Subpatch size:", args.subpatch_size, file=f)
        print("DNN model name:", args.model_name, file=f)
        print("Optimizer:", args.optimizer, file=f)
        print("Model architecture:\n", net, "\n", file=f)
        print("Device:", args.device)
        print("Batch size:", args.batch_size, file=f)
        print("Learning rate:", args.lr, file=f)
        print("Number of epochs:", args.epochs, file=f)
        print("Checkpoint every:", args.checkpoint_every, file=f)
        print("Normalize images:", args.normalize, file=f)
        print("Add checkerboard channel:", args.checkerboard, file=f)
    
    # Print train session info in terminal
    with open(traininfo_path, mode='r') as f:
        for line in f:
            print(line, end='')
    print("\n\n")

    train_loop(net, optimizer, train_loader, val_loader, args.epochs, output_dir,
               args.checkpoint_every)

    tmetrics = test_loop(net, test_loader)
    with open(os.path.join(output_dir, "test-results.json"), mode='w') as f:
        json.dump(tmetrics, f)

    print("Done!")
