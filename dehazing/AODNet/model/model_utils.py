import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage import feature
from torchvision.transforms.functional import rgb_to_grayscale


def gaussian_2d(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def coord_pos_norm(grid):
    # grid: [-1, 1]
    grid = (grid + 1) / 2
    grid = torch.clamp(grid, 0, 1)
    return grid


def coord_neg_norm(grid):
    # grid: [0, 1]
    grid = grid * 2 - 1
    grid = torch.clamp(grid, -1, 1)
    return grid


def grid_offset(coord, offset):
    # grid: b, 2, h, w
    x_cor = coord[:, 0, :, :]
    y_cor = coord[:, 1, :, :]
    x_cor += offset[:, :1, :, :]
    y_cor += offset[:, 1:, :, :]

    offseted_coord = torch.cat([x_cor, y_cor], dim=1)

    return offseted_coord


def invert(grid):  # h, w, 2
    I = np.zeros_like(grid)
    I[:, :, 1], I[:, :, 0] = np.indices((grid.shape[0], grid.shape[1]))
    P = np.copy(I)
    for i in range(5):
        P += (I - cv2.remap(grid, P, None, interpolation=cv2.INTER_LINEAR)) * 0.5
    return P
