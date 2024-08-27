import logging
import os

import jpeglib
import numpy as np
import torch
import torchvision

from config import default_config, get_config
from builders import build_chroma_net, build_spectral_net, build_idct, get_dct_stats
from jpegutils import (
    JPEGTransforms,
    SUBSAMPLE_FACTORS
)

from dataset import (
    ToDCTTensor,
    ToQTTensor,
)
from models import (
    SpectralModel,
    ConvertYccToRGB
)
from utils import load_checkpoint


def predict_spectral(spectral_luma, spectral_chroma, batch,
                     output_transform, subsample: int):
    assert output_transform in ("identity", "rgb", "ycc")

    y, cb, cr = batch["lq_dct_y"], batch["lq_dct_cb"], batch["lq_dct_cr"]
    qt_y, qt_c = batch["qt_y"], batch["qt_c"]

    y_pred  = spectral_luma(y, qt_y)
    cb_pred = spectral_chroma(cb, qt_c)
    cr_pred = spectral_chroma(cr, qt_c)

    # to_dct_tensor = ToDCTTensor(**dct_stats)

    # yy = to_dct_tensor(dctobj.Y, chroma=False).unsqueeze(0).to(device)
    # cb = to_dct_tensor(dctobj.Cb, chroma=True).unsqueeze(0).to(device)
    # cr = to_dct_tensor(dctobj.Cr, chroma=True).unsqueeze(0).to(device)

    result = {"dctY": y_pred, "dctCb": cb_pred, "dctCr": cr_pred}

    if output_transform != "identity":
        scale = SUBSAMPLE_FACTORS[subsample]
        # Upsample chroma components
        upsampled_cb = torch.nn.functional.interpolate(
            cb_pred, scale_factor=scale, mode="nearest"
        )
        upsampled_cr = torch.nn.functional.interpolate(
            cr_pred, scale_factor=scale, mode="nearest"
        )
        # Concatenate Y, Cb+, Cr+ planes
        ycc = torch.concatenate([y_pred, upsampled_cb, upsampled_cr], dim=1)
        if output_transform == "ycc":
            result["ycc"] = ycc
        else:
            result["tgb"] = ConvertYccToRGB()(ycc)
    return result


@torch.no_grad()
def deartifact_jpeg(config, spectral_luma, spectral_chroma, filepath: str):

    device = torch.device(config.TRAIN.DEVICE)

    dctobj = jpeglib.read_dct(filepath)
    H, W = dctobj.height, dctobj.width
    qt_y, qt_c = dctobj.qt[0], dctobj.qt[1]
    qt_y = ToQTTensor()(qt_y).to(device)
    qt_c = ToQTTensor()(qt_c).to(device)

    dct_stats = get_dct_stats(config)
    to_dct_tensor = ToDCTTensor(**dct_stats)

    yy = to_dct_tensor(dctobj.Y, chroma=False).unsqueeze(0).to(device)
    cb = to_dct_tensor(dctobj.Cb, chroma=True).unsqueeze(0).to(device)
    cr = to_dct_tensor(dctobj.Cr, chroma=True).unsqueeze(0).to(device)

    dct_yy_pred = spectral_luma(yy, qt_y)
    dct_cb_pred = spectral_chroma(cb, qt_c)
    dct_cr_pred = spectral_chroma(cr, qt_c)

    idct = build_idct(config).to(device=device)
    Y = idct(dct_yy_pred, chroma=False)
    Cb = idct(dct_cb_pred, chroma=True)
    Cr = idct(dct_cr_pred, chroma=True)

    # Upsample chroma components
    scale = tuple(dctobj.samp_factor[0])
    upsampled_cb = torch.nn.functional.interpolate(
        Cb, scale_factor=scale, mode="nearest"
    )
    upsampled_cr = torch.nn.functional.interpolate(
        Cr, scale_factor=scale, mode="nearest"
    )
    # Crop
    Y = Y[:, :, :H, :W]
    upsampled_cb = upsampled_cb[:, :, :H, :W]
    upsampled_cr = upsampled_cr[:, :, :H, :W]

    torgb = ConvertYccToRGB().to(device)
    ycc = torch.cat([Y, upsampled_cb, upsampled_cr], dim=1)
    rgb = torch.clamp(torgb(ycc), min=0.0, max=1.0)
    img = torchvision.transforms.ConvertImageDtype(torch.uint8)(rgb[0])
    return img.cpu()


def predict_test(dct_planes, qt, spectral_net, chroma_net, subsample):

    y, cb, cr = dct_planes
    qt_y, qt_c = qt

    luma_mean   = getattr(spectral_net.output_transform, "luma_mean", None)
    luma_std    = getattr(spectral_net.output_transform, "luma_std", None)
    chroma_mean = getattr(spectral_net.output_transform, "chroma_mean", None)
    chroma_std  = getattr(spectral_net.output_transform, "chroma_std", None)

    dct_transform = ToDCTTensor(luma_mean, luma_std, chroma_mean, chroma_std)
    y = dct_transform(y, chroma=False).unsqueeze(0)
    cb = dct_transform(cb, chroma=True).unsqueeze(0)
    cr = dct_transform(cr, chroma=True).unsqueeze(0)

    qt_transform = ToQTTensor()

    qt_y = qt_transform(qt[0])
    qt_c = qt_transform(qt[1])

    y_enhanced = spectral_net(y, qt_y, chroma=False)
    cb_enhanced = spectral_net(cb, qt_c, chroma=True)
    cr_enhanced = spectral_net(cr, qt_c, chroma=True)

    scale = SUBSAMPLE_FACTORS[subsample]

    h, w = y_enhanced.shape[-2], y_enhanced.shape[-1]
    # Upsample chroma components
    upsampled_cb = torch.nn.functional.interpolate(
        cb_enhanced, scale_factor=scale, mode="nearest"
    )
    upsampled_cr = torch.nn.functional.interpolate(
        cr_enhanced, scale_factor=scale, mode="nearest"
    )

    # Concatenate Y, Cb+, Cr+ planes
    ycc = torch.concatenate([y_enhanced, upsampled_cb[:,:, :h, :w], upsampled_cr[:,:, :h, :w]], dim=1)
    enhanced = chroma_net(ycc)
    assert ycc.shape[1] == 3

    return {"Y": y_enhanced[0], "Cb": cb_enhanced[0], "Cr": cr_enhanced[0], "final": enhanced[0]}



if __name__ == "__main__":

    # imgpath = "/Users/stefan/code/jpeg-deblock/data/BSDS500/BSDS500/data/images/test/108004.jpg"
    imgpath = "/Users/stefan/code/jpeg-deblock/data/overfit/bikes.png"

    config_path = "configs/small-v5-overfit.yaml"
    # dctobj = jpeglib.read_dct(imgpath)
    # rgb = jpeglib.read_spatial(imgpath).load()
    rgb = torchvision.io.read_image(imgpath).permute(1,2,0).numpy()

    # Write reference Y, Cb, Cr and RGB
    jpeg = JPEGTransforms(rgb)
    ycc = jpeg.get_ycc_planes(subsample=444)
    torchvision.io.write_png(torch.from_numpy(rgb.transpose((2,0,1))), "tests/bikes_final_ref.png")
    torchvision.io.write_png(torch.from_numpy(ycc[0]).unsqueeze(0), "tests/bikes_Y_ref.png")
    torchvision.io.write_png(torch.from_numpy(ycc[1]).unsqueeze(0), "tests/bikes_Cb_ref.png")
    torchvision.io.write_png(torch.from_numpy(ycc[2]).unsqueeze(0), "tests/bikes_Cr_ref.png")

    config = get_config(config_path)
    chroma_net = build_chroma_net(config)
    spectral_net = build_spectral_net(config)

    logger = logging.getLogger("null")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    config.defrost()
    config.MODEL.RESUME = "checkpoints/small-spectral-v5-overfit/checkpoint_1100.pth"
    state = {
        "spectral": spectral_net,
        "chroma": chroma_net,
        "iteration": config.TRAIN.NUM_ITERATIONS,
    }
    load_checkpoint(state, config, logger)
    spectral_net = state["spectral"]
    chroma_net = state["chroma"]

    # dct_planes = [dctobj.Y, dctobj.Cb, dctobj.Cr]
    dct_planes = jpeg.encode(quality=15, subsample=444)
    quantized_rgb = jpeg.decode_rgb(dct_planes, 444)
    qt = [jpeg.get_y_qtable(quality=15), jpeg.get_c_qtable(quality=15)]
    # qt = [dctobj.qt[0], dctobj.qt[1]]

    result = predict_test(dct_planes, qt, spectral_net, chroma_net, 444)
    outT = torchvision.transforms.ConvertImageDtype(torch.uint8)

    torchvision.io.write_png(outT(result["Y"]), filename="tests/bikes_Y_v5.png")
    torchvision.io.write_png(outT(result["Cb"]), filename="tests/bikes_Cb_v5.png")
    torchvision.io.write_png(outT(result["Cr"]), filename="tests/bikes_Cr_v5.png")
    torchvision.io.write_png(outT(result["final"]), filename="tests/bikes_final_v5.png")
    torchvision.io.write_png(outT(torch.from_numpy(quantized_rgb.transpose((2,0,1)))), filename="tests/bikes_q15.png")