import itertools
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision
import time
from PIL import Image
from timeit import timeit, Timer

from models import (
    ToDCTTensor,
    InverseDCT,
    ConvertYccToRGB,
    Local2DAttentionLayer,
    SpectralEncoderLayer,
    SpectralEncoder,
)
from jpegutils import JPEGTransforms, rgb2ycc, ycc2rgb
from utils import is_image


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DCT_STATS_FILEPATH = "data/DIV2K-DCT-coeff-stats.json"
TEST_IMAGES_DIR = "data/testimgs/"
TEST_IMAGES_PATHS = [x.path for x in os.scandir(TEST_IMAGES_DIR) if is_image(x.name)]


@pytest.mark.parametrize("batch_size, imsize, window_size",
                         [(1,8,3), (16, 8, 7), (32, 16, 5)])
def test_LocalAttentionBlock(
        batch_size, imsize, window_size, embed_dim=128, num_heads=4,
        device="cpu"
    ):

    device = torch.device(device)

    in_shape = (batch_size, embed_dim, imsize, imsize)
    out_shape = (batch_size, embed_dim, imsize, imsize)
    attn_shape = (batch_size, imsize**2, num_heads, 1, window_size ** 2)
    sample_input = torch.randn(in_shape, dtype=torch.float32).to(device=device)
    attn_block = Local2DAttentionLayer(window_size, embed_dim, num_heads)
    attn_block.to(device=device)

    output, attn_scores = attn_block(sample_input)
    assert output.shape == out_shape
    assert attn_scores.shape == attn_shape
    loss = torch.sum(output)
    loss.backward()


def test_LocalTransformerEncoder(
        batch_size=32, imsize=8, d_qcoeff=64,
        kernel_size=7, d_model=128, num_heads=4, d_feedforward=512, device="cpu"
    ):

    device = torch.device(device)

    img_shape = (batch_size, d_model, imsize, imsize)
    qct_shape = (d_qcoeff,)
    out_shape = (batch_size, d_model, imsize, imsize)

    sample_image = torch.randn(img_shape, dtype=torch.float32).to(device=device)
    sample_coeff = torch.randn(qct_shape, dtype=torch.float32).to(device=device)

    encoder = SpectralEncoderLayer(
        kernel_size, d_model, d_qcoeff, num_heads, d_feedforward
    ).to(device=device)

    elapsed = []
    for _ in range(10):
        tic = time.time()
        output = encoder(sample_image, sample_coeff)
        assert output.shape == out_shape
        loss = torch.sum(output)
        loss.backward()
        toc = time.time()
        elapsed.append(toc - tic)
    return float(np.mean(elapsed)), float(np.std(elapsed))


def test_DctTransformer(
        batch_size=64, in_features=64, num_encoders=4, kernel_size=7,
        d_model=128, d_qcoeff=64, imsize=8, num_heads=4, d_feedforward=512, device="cpu"
    ):

    device = torch.device(device)

    img_shape = (batch_size, in_features, imsize, imsize)
    qct_shape = (d_qcoeff)
    out_shape = (batch_size, d_model, imsize, imsize)

    sample_image = torch.randn(img_shape, dtype=torch.float32).to(device=device)
    sample_coeff = torch.randn(qct_shape, dtype=torch.float32).to(device=device)

    transformer = SpectralEncoder(
        in_features, num_encoders, kernel_size, d_model, d_qcoeff, num_heads,
        d_feedforward
    ).to(device=device)

    elapsed = []
    for _ in range(10):
        tic = time.time()
        output = transformer(sample_image, sample_coeff)
        assert output.shape == out_shape
        loss = torch.sum(output)
        loss.backward()
        toc = time.time()
        elapsed.append(toc - tic)
    return float(np.mean(elapsed)), float(np.std(elapsed))


@pytest.mark.parametrize("filepath, normalize",
                         list(itertools.product(TEST_IMAGES_PATHS, (True, False))))
def test_InverseDCT(filepath, normalize):

    # Read DCT coefficient stats
    if normalize:
        with open(DCT_STATS_FILEPATH, mode="r") as f:
            stats = json.load(f)
        luma_mean =     torch.as_tensor(stats["dct_Y_mean"])
        luma_std =      torch.as_tensor(stats["dct_Y_std"])
        chroma_mean =   torch.as_tensor(stats["dct_C_mean"])
        chroma_std =    torch.as_tensor(stats["dct_C_std"])
    else:
        luma_mean = None
        luma_std  = None
        chroma_mean = None
        chroma_std = None

    # Read image
    rgb = torchvision.transforms.PILToTensor()(Image.open(filepath))
    ycc = rgb2ycc(rgb.permute(1,2,0).numpy())
    jpegT = JPEGTransforms(rgb.permute(1,2,0).numpy())
    h, w = jpegT.height, jpegT.width
    dct = jpegT.get_dct_planes(subsample=444)

    # Initialize transforms
    to_dct_tensor = ToDCTTensor(luma_mean, luma_std, chroma_mean, chroma_std)
    to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)
    to_rgb = ConvertYccToRGB()
    idct = InverseDCT(luma_mean, luma_std, chroma_mean, chroma_std)

    dct_y_10 = to_dct_tensor(dct[0], chroma=False)
    dct_cb_10 = to_dct_tensor(dct[1], chroma=True)
    dct_cr_10 = to_dct_tensor(dct[2], chroma=True)

    assert dct_y_10.ndim == 3 and dct_y_10.shape[0] == 64
    assert dct_cb_10.ndim == 3 and dct_cb_10.shape[0] == 64

    # Add batch dimension
    dct_y_10 = dct_y_10.unsqueeze(0)
    dct_cb_10 = dct_cb_10.unsqueeze(0)
    dct_cr_10 = dct_cr_10.unsqueeze(0)

    # Convert and test Y Plane
    converted_y = idct(dct_y_10, chroma=False)
    converted_y = torch.clamp(converted_y, 0.0, 1.0)
    converted_y = to_uint8(converted_y).squeeze(dim=0).squeeze(dim=0)
    assert converted_y.ndim == 2
    converted_y = converted_y[:h, :w].numpy()
    true_y = ycc[:,:,0]
    assert np.all(converted_y == true_y)

    # Convert and test Cb Plane
    converted_cb = idct(dct_cb_10, chroma=True)
    converted_cb = torch.clamp(converted_cb, 0.0, 1.0)
    converted_cb = to_uint8(converted_cb).squeeze(dim=0).squeeze(dim=0)
    assert converted_cb.ndim == 2
    converted_cb = converted_cb[:h, :w].numpy()
    true_cb = ycc[:,:,1]
    assert np.all(converted_cb == true_cb)

    # Convert and test Cr Plane
    converted_cr = idct(dct_cr_10, chroma=True)
    converted_cr = torch.clamp(converted_cr, 0.0, 1.0)
    converted_cr = to_uint8(converted_cr).squeeze(dim=0).squeeze(dim=0)
    assert converted_cr.ndim == 2
    converted_cr = converted_cr[:h, :w].numpy()
    true_cr = ycc[:,:,2]
    assert np.all(converted_cr == true_cr)


@pytest.mark.parametrize("filepath", TEST_IMAGES_PATHS)
def test_ConvertYccToRGB(filepath):

    # Read image
    rgb = torchvision.transforms.PILToTensor()(Image.open(filepath))
    ycc = rgb2ycc(rgb.permute(1,2,0).numpy())
    jpegT = JPEGTransforms(rgb.permute(1,2,0).numpy())
    h, w = jpegT.height, jpegT.width
    dct = jpegT.get_dct_planes(subsample=444)

    # Initialize transforms
    to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)
    to_rgb = ConvertYccToRGB()

    ycc10 = torchvision.transforms.ToTensor()(ycc)
    rgb10 = to_uint8(to_rgb(ycc10).clip(0.0, 1.0))
    maxdiff = torch.max(torch.abs(rgb10.int() - rgb.int()))
    assert maxdiff <= 2
    # assert torch.all(torch.abs(rgb10.int() - rgb.int()) < 2)



if __name__ == "__main__":

    # # Test LocalAttentionBlock
    # kernels = (5, 7)
    # batches = (32, 64)
    # imsizes = (8, 10)
    # dmodels = (64, 128)

    # print("-" * 80, "\n\nBechmarking LocalAttentionBlock()...\n\n")
    # for K, N, W, E in itertools.product(kernels, batches, imsizes, dmodels):
    #     res = timeit("test_LocalAttentionBlock(batch_size=N, imsize=W, kernel_size=K, embed_dim=E, device=\"mps\")",
    #            number=10,
    #            globals=globals()
    #     )
    #     print(f"Time for (kernel_size={K}, batch_size={N}, imsize={W}, embed_dim={E}) config: {res:.3f}")

    # # Test LocalTransformerEncoder
    # kernels = (5, 7)
    # batches = (32, 64)
    # num_heads = (2, 4)
    # dense_sizes = (256, 512, 1024)
    # dmodels = (64, 128)

    # print("-" * 80, "\n\nBechmarking LocalTransformerEncoder()...\n\n")
    # params_iter = itertools.product(kernels, batches, num_heads, dmodels, dense_sizes)
    # for K, N, H, E, F in params_iter:
    #     mean, std = test_LocalTransformerEncoder(
    #         batch_size=N, kernel_size=K, d_model=E, d_feedforward=F, num_heads=H,
    #         device="mps"
    #     )
    #     print(f"Time for (kernel_size={K}, batch_size={N}, num_heads={H}, d_model={E}, d_feedforward={F}) config: {mean:.3f}s ± {std:.3f}")


    # Test DctTransformer
    batches = (32, 64)
    num_encoders = (2, 4, 8)
    num_heads = (2, 4)
    dense_sizes = (256, 512, 1024)
    dmodels = (64, 128)

    print("-" * 80, "\n\nBechmarking DctTransformer()...\n\n")
    params_iter = itertools.product(batches, num_encoders, num_heads, dmodels, dense_sizes)
    for N, E, H, D, F in params_iter:
        mean, std = test_DctTransformer(
            batch_size=N, num_encoders=E, d_model=D, d_feedforward=F, num_heads=H,
            device="mps"
        )
        print(f"Time for (num_encoders={E}, batch_size={N}, num_heads={H}, d_model={D}, d_feedforward={F}) config: {mean:.3f}s ± {std:.3f}")
