import numpy as np
import torch
import torchvision
import torchvision.transforms.functional

from models.transforms import ConvertRGBToYcc


def calculate_psnr(img1, img2, channel_dim):
    mse = calculate_mse(img1, img2, channel_dim)
    return -10.0 * np.log10(mse)


def calculate_mse(img1, img2, channel_dim):
    img1 = _convert_image(img1, channel_dim)
    img2 = _convert_image(img2, channel_dim)
    return torch.mean((img1 - img2) ** 2).item()


def calculate_ssim(img1, img2, channel_dim):
    img1 = _convert_image(img1, channel_dim)
    img2 = _convert_image(img2, channel_dim)

    nc = img1.shape[0]
    total = 0.0
    for c in range(nc):
        total += _ssim(img1[c:c+1], img2[c:c+1])

    return total / nc


def _ssim(img1, img2):
    assert img1.ndim == 3
    assert img2.ndim == 3
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    C1 = 0.01**2
    C2 = 0.03**2

    filter = torch.ones(1, 1, 8, 8) / 64

    mu_1 = torch.nn.functional.conv2d(img1, filter, bias=None)
    mu_2 = torch.nn.functional.conv2d(img2, filter, bias=None)

    var_1 = torch.nn.functional.conv2d(img1**2, filter) - mu_1**2
    var_2 = torch.nn.functional.conv2d(img2**2, filter) - mu_2**2
    cov_12 = torch.nn.functional.conv2d(img1*img2, filter) - mu_1 * mu_2

    ssim_blocks = ((2 * mu_1 * mu_2 + C1) * (2 * cov_12 + C2)) / ((mu_1**2 + mu_2**2 + C1) * (var_1 + var_2 + C2))
    return ssim_blocks.mean().item()


def calculate_psnrb(img1, img2, channel_dim):
    img1 = _convert_image(img1, channel_dim)
    img2 = _convert_image(img2, channel_dim)

    # I don't know why, but Block Effect Factor is computed only for Y channel
    if img1.ndim == 3:
        bef = compute_bef(ConvertRGBToYcc()(img1.unsqueeze(0))[0,0])
    else:
        bef = compute_bef(img1[0])
    mse = torch.mean((img1 - img2)**2).item()
    mse_b = mse + bef
    if mse_b == 0:
        return float('inf')
    return -10 * np.log10(mse_b)


def compute_bef(img: torch.Tensor):
    assert img.ndim == 2
    height, width = img.shape
    block = 8

    H = [i for i in range(width-1)]
    H_B = [i for i in range(block-1,width-1,block)]
    H_BC = list(set(H)-set(H_B))

    V = [i for i in range(height-1)]
    V_B = [i for i in range(block-1,height-1,block)]
    V_BC = list(set(V)-set(V_B))

    D_B = torch.zeros(1, dtype=torch.float32)
    D_BC = torch.zeros(1, dtype=torch.float32)

    for i in H_B:
        diff = img[:,i] - img[:,i+1]
        D_B += torch.sum(diff**2)

    for i in H_BC:
        diff = img[:,i] - img[:,i+1]
        D_BC += torch.sum(diff**2)


    for j in V_B:
        diff = img[j,:] - img[j+1,:]
        D_B += torch.sum(diff**2)

    for j in V_BC:
        diff = img[j,:] - img[j+1,:]
        D_BC += torch.sum(diff**2)

    N_HB = height * (width/block - 1)
    N_HBC = height * (width - 1) - N_HB
    N_VB = width * (height/block -1)
    N_VBC = width * (height -1) - N_VB
    D_B = D_B / (N_HB + N_VB)
    D_BC = D_BC / (N_HBC + N_VBC)
    eta = np.log2(block) / np.log2(min(height, width)) if D_B > D_BC else 0
    return eta * (D_B - D_BC).item()


def _convert_image(img, channel_dim="first"):
    img = torch.as_tensor(img)
    assert img.ndim == 3
    if channel_dim == "last":
        img = img.permute(2,0,1)
    x = torchvision.transforms.functional.convert_image_dtype(img, torch.float32)
    return x
