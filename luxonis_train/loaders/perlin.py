import random
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch


def compute_gradients(res: tuple[int, int]) -> torch.Tensor:
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    return gradients


@torch.jit.script
def lerp_torch(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    return (y - x) * w + x


def fade_function(t: torch.Tensor) -> torch.Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def tile_grads(
    slice1: Tuple[int, int | None],
    slice2: Tuple[int, int | None],
    gradients: torch.Tensor,
    d: Tuple[int, int],
) -> torch.Tensor:
    return (
        gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )


def dot(
    grad: torch.Tensor,
    shift: Tuple[int, int],
    grid: torch.Tensor,
    shape: Tuple[int, int],
) -> torch.Tensor:
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


def rand_perlin_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    fade: Callable[[torch.Tensor], torch.Tensor] = fade_function,
) -> torch.Tensor:
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, res[0], delta[0]),
        torch.arange(0, res[1], delta[1]),
        indexing="ij",
    )
    grid = torch.stack((grid_x % 1, grid_y % 1), dim=-1)

    gradients = compute_gradients(res)

    n00 = dot(tile_grads((0, -1), (0, -1), gradients, d), (0, 0), grid, shape)
    n10 = dot(
        tile_grads((1, None), (0, -1), gradients, d), (-1, 0), grid, shape
    )
    n01 = dot(
        tile_grads((0, -1), (1, None), gradients, d), (0, -1), grid, shape
    )
    n11 = dot(
        tile_grads((1, None), (1, None), gradients, d), (-1, -1), grid, shape
    )

    t = fade(grid[: shape[0], : shape[1]])

    return torch.sqrt(torch.tensor(2.0)) * lerp_torch(
        lerp_torch(n00, n10, t[..., 0]),
        lerp_torch(n01, n11, t[..., 0]),
        t[..., 1],
    )


@torch.jit.script
def rotate_noise(noise: torch.Tensor) -> torch.Tensor:
    angle = torch.rand(1) * 2 * torch.pi
    h, w = noise.shape
    center_y, center_x = h // 2, w // 2
    y, x = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
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
    shape: Tuple[int, int],
    min_perlin_scale: int = 0,
    perlin_scale: int = 6,
    threshold: float = 0.5,
) -> torch.Tensor:
    perlin_scalex = 2 ** int(
        torch.randint(min_perlin_scale, perlin_scale, (1,)).item()
    )
    perlin_scaley = 2 ** int(
        torch.randint(min_perlin_scale, perlin_scale, (1,)).item()
    )
    perlin_noise = rand_perlin_2d(
        shape=shape, res=(perlin_scalex, perlin_scaley)
    )
    perlin_mask = torch.where(
        perlin_noise > threshold,
        torch.ones_like(perlin_noise, dtype=torch.float32),
        torch.zeros_like(perlin_noise, dtype=torch.float32),
    )
    perlin_mask = rotate_noise(perlin_mask)
    return perlin_mask


def load_image_as_numpy(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = image.astype(np.float32) / 255.0
    return image


def apply_anomaly_to_img(
    img: torch.Tensor,
    anomaly_source_paths: List[str],
    beta: float | None = None,
    pixel_augs: List[Callable] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Perlin noise-based anomalies to a single image (C, H, W).

    @type img: torch.Tensor
    @param img: The input image tensor of shape (C, H, W).
    @type anomaly_source_paths: List[str]
    @param anomaly_source_paths: List of file paths to the anomaly images.
    @type pixel_augs: List[Callable] | None
    @param pixel_augs: A list of albumentations augmentations to apply to the anomaly image. Defaults to C{None}.
    @type beta: float | None
    @param beta: A blending factor for anomaly and noise. If None, a random value in the range [0, 0.8]
                 is used. Defaults to C{None}.
    @rtype: Tuple[torch.Tensor, torch.Tensor]
    @return: A tuple containing:
        - augmented_img (torch.Tensor): The augmented image with applied anomaly and Perlin noise.
        - perlin_mask (torch.Tensor): The Perlin noise mask applied to the image.
    """

    if pixel_augs is None:
        pixel_augs = []

    sampled_anomaly_image_path = random.choice(anomaly_source_paths)

    anomaly_image = load_image_as_numpy(sampled_anomaly_image_path)

    anomaly_image = cv2.resize(
        anomaly_image,
        (img.shape[2], img.shape[1]),
        interpolation=cv2.INTER_LINEAR,
    )

    for aug in pixel_augs:
        anomaly_image = aug(image=anomaly_image)["image"]

    anomaly_image = torch.tensor(anomaly_image).permute(2, 0, 1)

    perlin_mask = generate_perlin_noise(
        shape=(img.shape[1], img.shape[2]),
    )

    if beta is None:
        beta = torch.rand(1).item() * 0.8

    augmented_img = (
        (1 - perlin_mask).unsqueeze(0) * img
        + (1 - beta) * perlin_mask.unsqueeze(0) * anomaly_image
        + beta * perlin_mask.unsqueeze(0) * img
    )

    return augmented_img, perlin_mask
