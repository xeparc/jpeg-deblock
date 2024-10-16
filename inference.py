from collections import namedtuple

import numpy as np
import torch
import torchvision
import jpeglib
import torchvision.transforms.functional
from scipy.fft import idctn

from models.transforms import ToDCTTensor, ToQTTensor


FlareInputs = namedtuple(
    "FlareInputs",
    ["y", "cb", "cr", "dct_y", "dct_cb", "dct_cr", "qt_y", "qt_c", "height", "width"]
)


def inverse_dct(dct_blocks: np.ndarray):
    plane = 128.0 + idctn(dct_blocks, type=2, axes=(2,3), norm="ortho")
    plane = np.ascontiguousarray(np.transpose(plane, (0,2,1,3)))
    plane = plane.reshape(8 * plane.shape[0], 8 * plane.shape[2])
    plane = np.clip(np.round(plane), a_min=0, a_max=255)
    return (plane / 255.0).astype(np.float32)


def read_inputs(filepath: str) -> FlareInputs:
    dctobj = jpeglib.read_dct(filepath)
    # Get height & width
    h, w = dctobj.height, dctobj.width

    if dctobj.has_chrominance:
        # Get quantization tables
        if len(dctobj.qt) == 3:
            qt_y  = dctobj.qt[0]
            qt_cb = dctobj.qt[1]
            qt_cr = dctobj.qt[2]
        else:
            qt_y  = dctobj.qt[0]
            qt_cb = dctobj.qt[1]
            qt_cr = dctobj.qt[1]
        # Get DCT coefficients
        dct_y  = dctobj.Y * qt_y
        dct_cb = dctobj.Cb * qt_cb
        dct_cr = dctobj.Cr * qt_cr
        # Get channels
        y  = inverse_dct(dct_y)
        cb = inverse_dct(dct_cb)
        cr = inverse_dct(dct_cr)

        return FlareInputs(
            y=      torch.from_numpy(y).view(1, y.shape[0], y.shape[1]),
            cb=     torch.from_numpy(cb).view(1, cb.shape[0], cb.shape[1]),
            cr=     torch.from_numpy(cr).view(1, cr.shape[0], cr.shape[1]),
            dct_y=  torch.from_numpy(dct_y),
            dct_cb= torch.from_numpy(dct_cb),
            dct_cr= torch.from_numpy(dct_cr),
            qt_y=   torch.from_numpy(qt_y),
            qt_c=   torch.from_numpy(qt_cb),
            height= h,
            width=  w
        )

    else:
        # Get quantization table
        qt_y  = dctobj.qt[0]
        # Get DCT coefficients
        dct_y  = dctobj.Y * qt_y
        # Get channels
        y  = inverse_dct(dct_y)

        return FlareInputs(
            y=      torch.from_numpy(y).view(1, y.shape[0], y.shape[1]),
            cb=     None,
            cr=     None,
            dct_y=  torch.from_numpy(dct_y),
            dct_cb= None,
            dct_cr= None,
            qt_y=   torch.from_numpy(qt_y),
            qt_c=   None,
            height= h,
            width=  w
        )


@torch.no_grad()
def enhance_grayscale(filepath: str, model: torch.nn.Module, output_dtype: torch.dtype):

    inputs = read_inputs(filepath)

    dct_transform = ToDCTTensor(
            luma_mean=      model.idct.luma_mean,
            luma_std=       model.idct.luma_std,
            chroma_mean=    model.idct.chroma_mean,
            chroma_std=     model.idct.chroma_std
    )
    qt_transform = ToQTTensor(invert=False)
    to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)

    y = inputs.y
    dct_y = dct_transform(inputs.dct_y, chroma=False)
    qt_y = qt_transform(inputs.qt_y)

    enhanced = model(y.unsqueeze(0), dct_y.unsqueeze(0), qt_y.unsqueeze(0))
    enhanced = enhanced.cpu()[0]
    enhanced = torch.clip(enhanced, 0.0, 1.0)
    # Crop
    enhanced = enhanced[:, :inputs.height, :inputs.width]

    if output_dtype == torch.float32:
        return enhanced
    elif output_dtype == torch.uint8:
        return to_uint8(enhanced)
    else:
        raise NotImplementedError(output_dtype)


@torch.no_grad()
def enhance_color(filepath: str, model: torch.nn.Module, output_dtype: torch.dtype):

    inputs = read_inputs(filepath)
    dct_transform = ToDCTTensor(
            luma_mean=      model.chroma.idct.luma_mean,
            luma_std=       model.chroma.idct.luma_std,
            chroma_mean=    model.chroma.idct.chroma_mean,
            chroma_std=     model.chroma.idct.chroma_std
    )
    qt_transform = ToQTTensor(invert=False)
    to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)

    y  = inputs.y.unsqueeze(0)
    cb = inputs.cb.unsqueeze(0)
    cr = inputs.cr.unsqueeze(0)

    dct_y  = dct_transform(inputs.dct_y, chroma=False).unsqueeze(0)
    dct_cb = dct_transform(inputs.dct_cb, chroma=True).unsqueeze(0)
    dct_cr = dct_transform(inputs.dct_cr, chroma=True).unsqueeze(0)

    qt_y = qt_transform(inputs.qt_y).unsqueeze(0)
    qt_c = qt_transform(inputs.qt_c).unsqueeze(0)

    enhanced = model(y, cb, cr, dct_y, dct_cb, dct_cr, qt_y, qt_c)
    enhanced = enhanced.cpu()[0]
    enhanced = torch.clip(enhanced, 0.0, 1.0)
    # Crop
    enhanced = enhanced[:, :inputs.height, :inputs.width]

    if output_dtype == torch.float32:
        return enhanced
    elif output_dtype == torch.uint8:
        return to_uint8(enhanced)
    else:
        raise NotImplementedError(output_dtype)
