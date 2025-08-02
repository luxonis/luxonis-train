import math
from collections.abc import Callable

import torch
from torch import Tensor


def compute_gradients(res: tuple[int, int]) -> Tensor:
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


@torch.jit.script
def lerp_torch(  # pragma: no cover
    x: Tensor, y: Tensor, w: Tensor
) -> Tensor:
    return (y - x) * w + x


def fade_function(t: Tensor) -> Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def tile_grads(
    slice1: tuple[int, int | None],
    slice2: tuple[int, int | None],
    gradients: Tensor,
    d: tuple[int, int],
) -> Tensor:
    return (
        gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )


def dot(
    grad: Tensor, shift: tuple[int, int], grid: Tensor, shape: tuple[int, int]
) -> Tensor:
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
    shape: tuple[int, int],
    res: tuple[int, int],
    fade: Callable[[Tensor], Tensor] = fade_function,
) -> Tensor:
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

    return torch.tensor(math.sqrt(2.0)) * lerp_torch(
        lerp_torch(n00, n10, t[..., 0]),
        lerp_torch(n01, n11, t[..., 0]),
        t[..., 1],
    )


@torch.jit.script
def rotate_noise(noise: Tensor) -> Tensor:  # pragma: no cover
    angle = torch.rand(1) * 2 * torch.pi
    h, w = noise.shape
    center_y, center_x = h // 2, w // 2
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    x_shifted = x - center_x
    y_shifted = y - center_y
    cos_a = angle.cos()
    sin_a = angle.sin()
    rot_x = cos_a * x_shifted - sin_a * y_shifted + center_x
    rot_y = sin_a * x_shifted + cos_a * y_shifted + center_y
    rot_x = rot_x.clamp_(0, w - 1).long()
    rot_y = rot_y.clamp_(0, h - 1).long()
    return noise[rot_y, rot_x]


def generate_perlin_noise(
    shape: tuple[int, int],
    min_perlin_scale: int = 0,
    perlin_scale: int = 6,
    threshold: float = 0.5,
) -> Tensor:
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
    return rotate_noise(perlin_mask)


def apply_anomaly_to_img(
    img: Tensor, anomaly_img: Tensor, beta: float | None = None
) -> tuple[Tensor, Tensor]:
    """Applies Perlin noise-based anomalies to a single image (C, H, W).

    @type img: Tensor
    @param img: The input image tensor of shape (C, H, W).
    @type anomaly_source_paths: list[str]
    @param anomaly_source_paths: List of file paths to the anomaly images.
    @type pixel_augs: list[Callable] | None
    @param pixel_augs: A list of albumentations augmentations to apply to the anomaly image. Defaults to C{None}.
    @type beta: float | None
    @param beta: A blending factor for anomaly and noise. If None, a random value in the range [0, 0.8]
                 is used. Defaults to C{None}.
    @rtype: tuple[Tensor, Tensor]
    @return: A tuple containing:
        - augmented_img (Tensor): The augmented image with applied anomaly and Perlin noise.
        - perlin_mask (Tensor): The Perlin noise mask applied to the image.
    """
    perlin_mask = generate_perlin_noise(shape=(img.shape[1], img.shape[2]))

    if beta is None:
        beta = torch.rand(1).item() * 0.8

    augmented_img = (
        (1 - perlin_mask).unsqueeze(0) * img
        + (1 - beta) * perlin_mask.unsqueeze(0) * anomaly_img
        + beta * perlin_mask.unsqueeze(0) * img
    )

    return augmented_img, perlin_mask
