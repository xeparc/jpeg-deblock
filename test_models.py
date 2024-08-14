import itertools
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
import time
from timeit import timeit
from timeit import Timer

from models import (
    LocalAttentionBlock,
    LocalTransformerEncoder,
    DctTransformer
)


def test_LocalAttentionBlock(
        batch_size=64, imsize=8, kernel_size=7, embed_dim=128, num_heads=4,
        device="cpu"
    ):

    device = torch.device(device)

    in_shape = (batch_size, embed_dim, imsize, imsize)
    out_shape = (batch_size, embed_dim, imsize, imsize)
    sample_input = torch.randn(in_shape, dtype=torch.float32).to(device=device)
    attn_block = LocalAttentionBlock(kernel_size, embed_dim, num_heads)
    attn_block.to(device=device)

    output = attn_block(sample_input)
    assert output.shape == out_shape
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

    encoder = LocalTransformerEncoder(
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

    transformer = DctTransformer(
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
