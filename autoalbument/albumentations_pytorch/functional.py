import torch
import cv2
import numpy as np
from functools import wraps

from autoalbument.albumentations_pytorch.affine import (
    get_rotation_matrix,
    get_scaling_matrix,
    warp_affine,
)
from autoalbument.albumentations_pytorch.utils import (
    MAX_VALUES_BY_DTYPE,
    TorchPadding,
    clipped,
)


def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1


def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    @wraps(process_fn)
    def __process_fn(img):
        device = img.device
        img = img.data.cpu().numpy()
        img = process_fn(img, **kwargs)
        img = torch.tensor(img).to(device)  # .permute(2,0,1)
        return img

    return __process_fn


@clipped
def gaussian_blur(img_batch, gauss=0):
    blur_fn = _maybe_process_in_chunks(cv2.GaussianBlur, ksize=(3, 3), sigmaX=gauss)

    img_batch = img_batch.permute(0, 2, 3, 1)
    batch_result = img_batch.detach().clone()

    for idx, img in enumerate(img_batch):
        batch_result[idx] = blur_fn(img)

    return batch_result.permute(0, 3, 1, 2)


@clipped
def sharpen(img_batch, kernel):
    img_batch = img_batch.permute(0, 2, 3, 1)
    batch_result = img_batch.detach().clone()

    conv_fn = _maybe_process_in_chunks(cv2.filter2D, ddepth=-1, kernel=kernel)
    for idx, img in enumerate(img_batch):
        batch_result[idx] = conv_fn(img)

    return batch_result.permute(0, 3, 1, 2)


@clipped
def gauss_noise(img_batch, gauss):
    return img_batch + gauss


def solarize(img_batch, threshold):
    dtype = img_batch.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]
    return torch.where(img_batch >= threshold, max_val - img_batch, img_batch)


@clipped
def shift_rgb(img_batch, r_shift, g_shift, b_shift):
    result_img_batch = img_batch.clone()
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img_batch[:, i] = result_img_batch[:, i] + shift
    return result_img_batch


@clipped
def brightness_adjust(img_batch, beta):
    return img_batch + beta


@clipped
def contrast_adjust(img_batch, alpha):
    return img_batch * alpha


@clipped
def gaussian_blurr(img_batch, alpha):
    return gaussian_filter(a, sigma=alpha)


def vflip(img_batch):
    return torch.flip(img_batch, [-2]).contiguous()


def hflip(img_batch):
    return torch.flip(img_batch, [-1]).contiguous()


def shift_x(img_batch, dx, padding_mode=TorchPadding.REFLECTION):
    scaling_matrix = get_scaling_matrix(img_batch, dx=dx)
    return warp_affine(img_batch, scaling_matrix, padding_mode)


def shift_y(img_batch, dy, padding_mode=TorchPadding.REFLECTION):
    scaling_matrix = get_scaling_matrix(img_batch, dy=dy)
    return warp_affine(img_batch, scaling_matrix, padding_mode)


def rotate(img_batch, angle, padding_mode=TorchPadding.REFLECTION):
    rotation_matrix = get_rotation_matrix(img_batch, angle=angle)
    return warp_affine(img_batch, rotation_matrix, padding_mode)


def scale(img_batch, scale, padding_mode=TorchPadding.REFLECTION):
    rotation_matrix = get_rotation_matrix(img_batch, scale=scale)
    return warp_affine(img_batch, rotation_matrix, padding_mode)


def cutout(img_batch, num_holes, hole_size, fill_value=0):
    img_batch = img_batch.clone()
    height, width = img_batch.shape[-2:]
    for _n in range(num_holes):
        if height == hole_size:
            y1 = torch.tensor([0])
        else:
            y1 = torch.randint(0, height - hole_size, (1,))
        if width == hole_size:
            x1 = torch.tensor([0])
        else:
            x1 = torch.randint(0, width - hole_size, (1,))
        y2 = y1 + hole_size
        x2 = x1 + hole_size
        img_batch[:, :, y1:y2, x1:x2] = fill_value
    return img_batch

