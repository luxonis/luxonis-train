import glob
import os
import random
from functools import lru_cache

import cv2
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float16)


@lru_cache(maxsize=32)
def compute_gradients(res):
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    return gradients


@torch.jit.script
def lerp_torch(x, y, w):
    return (y - x) * w + x


def fade_function(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def tile_grads(slice1, slice2, gradients, d):
    return (
        gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )


def dot(grad, shift, grid, shape):
    return (
        torch.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            dim=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)


def rand_perlin_2d(shape, res, fade=fade_function):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, res[0], delta[0], device=device),
        torch.arange(0, res[1], delta[1], device=device),
        indexing="ij",
    )
    grid = torch.stack((grid_x % 1, grid_y % 1), dim=-1)

    gradients = compute_gradients(res)

    # Use the refactored tile_grads and dot functions
    n00 = dot(tile_grads([0, -1], [0, -1], gradients, d), [0, 0], grid, shape)
    n10 = dot(
        tile_grads([1, None], [0, -1], gradients, d), [-1, 0], grid, shape
    )
    n01 = dot(
        tile_grads([0, -1], [1, None], gradients, d), [0, -1], grid, shape
    )
    n11 = dot(
        tile_grads([1, None], [1, None], gradients, d), [-1, -1], grid, shape
    )

    t = fade(grid[: shape[0], : shape[1]])

    return torch.sqrt(torch.tensor(2.0, device=device)) * lerp_torch(
        lerp_torch(n00, n10, t[..., 0]),
        lerp_torch(n01, n11, t[..., 0]),
        t[..., 1],
    )


@torch.jit.script
def rotate_noise(noise):
    angle = torch.rand(1, device=noise.device) * 2 * torch.pi
    h, w = noise.shape
    center_y, center_x = h // 2, w // 2
    y, x = torch.meshgrid(
        torch.arange(h, device=noise.device),
        torch.arange(w, device=noise.device),
        indexing="ij",
    )
    x_shifted = x - center_x
    y_shifted = y - center_y
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rot_x = cos_a * x_shifted - sin_a * y_shifted + center_x
    rot_y = sin_a * x_shifted + cos_a * y_shifted + center_y
    rot_x = torch.clamp(rot_x, 0, w - 1).long()
    rot_y = torch.clamp(rot_y, 0, h - 1).long()
    return noise[rot_y, rot_x]


def generate_perlin_noise(
    shape, min_perlin_scale=0, perlin_scale=6, threshold=0.5
):
    perlin_scalex = 2 ** int(
        torch.randint(
            min_perlin_scale, perlin_scale, (1,), device=device
        ).item()
    )
    perlin_scaley = 2 ** int(
        torch.randint(
            min_perlin_scale, perlin_scale, (1,), device=device
        ).item()
    )
    perlin_noise = rand_perlin_2d(shape, (perlin_scalex, perlin_scaley))
    perlin_mask = torch.where(
        perlin_noise > threshold,
        torch.ones_like(perlin_noise),
        torch.zeros_like(perlin_noise),
    )
    perlin_mask = rotate_noise(perlin_mask)
    return perlin_mask


def load_image_as_tensor(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = torch.tensor(
        image.astype(np.float32) / 255.0, device=device
    ).permute(2, 0, 1)
    return image


def apply_anomaly_to_batch(batch, anomaly_source_path, beta=None):
    anomaly_source_paths = sorted(
        glob.glob(os.path.join(anomaly_source_path, "*/*.jpg"))
    )
    sampled_anomaly_image_path = random.choice(anomaly_source_paths)

    anomaly_image = load_image_as_tensor(sampled_anomaly_image_path)
    anomaly_image = torch.nn.functional.interpolate(
        anomaly_image.unsqueeze(0),
        size=batch.shape[2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    perlin_masks = [
        generate_perlin_noise((batch.shape[2], batch.shape[3]))
        for _ in range(batch.size(0))
    ]
    perlin_masks = torch.stack(perlin_masks).unsqueeze(1).to(device)
    perlin_masks = perlin_masks.expand_as(batch)

    if beta is None:
        beta = torch.rand(1, device=device).item() * 0.8

    # Apply anomaly to batch
    augmented_batch = (
        (1 - perlin_masks) * batch
        + (1 - beta) * perlin_masks * anomaly_image.unsqueeze(0)
        + beta * perlin_masks * batch
    )

    perlin_mask_bg = 1 - perlin_masks
    perlin_masks = torch.cat([perlin_masks, perlin_mask_bg], dim=1)

    return augmented_batch, perlin_masks
