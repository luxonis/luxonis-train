"""Perlin noise masks, and the blend that turns a texture into an
anomaly.

`LuxonisLoaderPerlinNoise` calls `apply_anomaly_to_img`, which draws its
mask with `generate_perlin_noise`. The other functions are the steps of
the noise.

"""

import math
from collections.abc import Callable

import torch
from torch import Tensor


def compute_gradients(res: tuple[int, int]) -> Tensor:
    r"""Draw a random unit gradient at each point of the lattice.

    The angles are uniform in :math:`[0, 2\pi)`. They use the global
    ``torch`` random state.

    Args:
        res (tuple[int, int]): The number of lattice cells along each
            dimension.

    Returns:
        ``Tensor``: The gradients ``(cos, sin)`` of shape
        ``[res[0] + 1, res[1] + 1, 2]``.

    Example:
        >>> import torch
        >>> gradients = compute_gradients((2, 3))
        >>> gradients.shape
        torch.Size([3, 4, 2])
        >>> torch.allclose(gradients.norm(dim=-1), torch.ones(3, 4))
        True

    """
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


@torch.jit.script
def lerp_torch(  # pragma: no cover
    x: Tensor, y: Tensor, w: Tensor
) -> Tensor:
    r"""Interpolate linearly from ``x`` to ``y`` with the weight ``w``.

    .. math::

        \operatorname{lerp}(x, y, w) = x + w (y - x)

    TorchScript compiles the function.

    Args:
        x (``Tensor``): The values at ``w = 0``.
        y (``Tensor``): The values at ``w = 1``.
        w (``Tensor``): The weights. They broadcast with ``x`` and ``y``.

    Returns:
        ``Tensor``: The interpolated values, with the broadcast shape of
        ``x``, ``y``, and ``w``.

    """
    return (y - x) * w + x


def fade_function(t: Tensor) -> Tensor:
    r"""Apply the quintic fade curve of Perlin noise.

    .. math::

        f(t) = 6 t^5 - 15 t^4 + 10 t^3

    The curve maps ``0`` to ``0`` and ``1`` to ``1``. Its first and
    second derivatives are zero at both ends. Thus the first and the
    second derivatives of the noise are continuous across the borders
    of the lattice cells.

    Args:
        t (``Tensor``): The positions inside a cell, in ``[0, 1]``.

    Returns:
        ``Tensor``: The faded positions, with the shape of ``t``.

    Example:
        >>> import torch
        >>> fade_function(torch.tensor([0.0, 0.25, 0.5, 1.0])).tolist()
        [0.0, 0.103515625, 0.5, 1.0]

    """
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def tile_grads(
    slice1: tuple[int, int | None],
    slice2: tuple[int, int | None],
    gradients: Tensor,
    d: tuple[int, int],
) -> Tensor:
    """Repeat one corner of the gradient lattice over the pixels of
    each cell.

    The function takes the slice
    ``gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]]``. Then it
    repeats each row ``d[0]`` times and each column ``d[1]`` times.
    `rand_perlin_2d` calls it once for each of the four cell corners.
    The slice ``(0, -1)`` selects the first corner along a dimension,
    and ``(1, None)`` selects the second corner.

    Args:
        slice1 (tuple[int, int | None]): The start and the stop of the
            slice along dimension ``0``.
        slice2 (tuple[int, int | None]): The start and the stop of the
            slice along dimension ``1``.
        gradients (``Tensor``): The lattice gradients. `rand_perlin_2d`
            gives the shape ``[res[0] + 1, res[1] + 1, 2]``.
        d (tuple[int, int]): The number of pixels in a cell along each
            dimension.

    Returns:
        ``Tensor``: The repeated gradients. For the slices of
        `rand_perlin_2d`, the shape is
        ``[res[0] * d[0], res[1] * d[1], 2]``.

    Example:
        A 2D tensor shows the pattern of the indices:

        >>> import torch
        >>> gradients = torch.arange(9).reshape(3, 3)
        >>> tile_grads((0, -1), (1, None), gradients, (2, 1)).tolist()
        [[1, 2], [1, 2], [4, 5], [4, 5]]

    """
    return (
        gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )


def dot(
    grad: Tensor, shift: tuple[int, int], grid: Tensor, shape: tuple[int, int]
) -> Tensor:
    """Compute the dot product of each pixel offset and its corner
    gradient.

    The offset from a cell corner to a pixel is ``grid + shift``. The
    function crops ``grid`` and ``grad`` to ``shape`` before the
    product.

    Args:
        grad (``Tensor``): The gradient of the corner for each pixel, of
            shape ``[H', W', 2]``, where ``H' >= H`` and ``W' >= W``.
        shift (tuple[int, int]): The negative position of the corner in
            the cell: ``(0, 0)``, ``(-1, 0)``, ``(0, -1)``, or
            ``(-1, -1)``.
        grid (``Tensor``): The position of each pixel inside its cell,
            in ``[0, 1)``, of shape ``[H'', W'', 2]``, where
            ``H'' >= H`` and ``W'' >= W``.
        shape (tuple[int, int]): The output shape ``(H, W)``.

    Returns:
        ``Tensor``: The dot products of shape ``[H, W]``.

    Example:
        A pixel at ``(0.25, 0.75)`` in its cell, and the corner
        ``(1, 0)`` with the gradient ``(1, 0)``:

        >>> import torch
        >>> grad = torch.tensor([[[1.0, 0.0]]])
        >>> grid = torch.tensor([[[0.25, 0.75]]])
        >>> dot(grad, (-1, 0), grid, (1, 1)).tolist()
        [[-0.75]]

    """
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
    r"""Generate 2D Perlin noise with random gradients.

    The function splits the output into ``res[0]`` by ``res[1]`` lattice
    cells. It draws a random gradient at each lattice point with
    `compute_gradients`. The value of a pixel blends the dot products of
    its four cell corners, with the weights from ``fade``. The factor
    :math:`\sqrt{2}` scales the values to ``[-1, 1]``. The noise is
    ``0`` at each lattice point.

    Each size in ``shape`` must be a multiple of the value in ``res`` for
    the same dimension. Otherwise a tensor operation raises
    ``RuntimeError``. A size of ``1`` with more than one cell gives an
    empty tensor instead.

    Args:
        shape (tuple[int, int]): The output shape ``(H, W)``.
        res (tuple[int, int]): The number of lattice cells along each
            dimension. More cells give smaller noise features.
        fade (``Callable[[Tensor], Tensor]``): The interpolation curve.
            It gets the position of each pixel inside its cell.

    Returns:
        ``Tensor``: The noise of shape ``[H, W]``, with values in
        ``[-1, 1]``.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> noise = rand_perlin_2d((8, 8), (2, 2))
        >>> noise.shape
        torch.Size([8, 8])
        >>> bool(noise.abs().max() <= 1)
        True
        >>> bool((noise[::4, ::4] == 0).all())
        True

    """
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
    r"""Rotate a 2D tensor by a random angle around its center.

    The angle is uniform in :math:`[0, 2\pi)` and uses the global
    ``torch`` random state. The center is ``(H // 2, W // 2)``. Each
    output pixel takes the value of the input pixel at the rotated
    position, with the coordinates rounded down. A position outside the
    input moves to the nearest border. TorchScript compiles the function.

    Args:
        noise (``Tensor``): The tensor of shape ``[H, W]``.

    Returns:
        ``Tensor``: The rotated tensor of shape ``[H, W]``. It holds only
        values from ``noise``.

    """
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
    """Generate a random binary mask from thresholded Perlin noise.

    The function draws an exponent ``k`` for each dimension, uniform
    over the integers in ``[min_perlin_scale, perlin_scale)``. The noise
    from `rand_perlin_2d` has ``2 ** k`` cells along that dimension. A
    pixel with noise above ``threshold`` gets ``1.0``, and the other
    pixels get ``0.0``. Then `rotate_noise` rotates the mask by a random
    angle.

    Each size in ``shape`` must be a multiple of
    ``2 ** (perlin_scale - 1)``, which is ``32`` for the default.
    Otherwise some draws fail, see `rand_perlin_2d`.

    Args:
        shape (tuple[int, int]): The mask shape ``(H, W)``.
        min_perlin_scale (int): The smallest exponent.
        perlin_scale (int): The upper bound of the exponent, exclusive.
            A larger exponent gives smaller noise blobs. A value that is
            not greater than ``min_perlin_scale`` makes ``torch.randint``
            raise ``RuntimeError``.
        threshold (float): The noise value that a pixel must exceed to
            get ``1.0``. The noise is in ``[-1, 1]``, so a higher
            threshold gives a smaller mask area.

    Returns:
        ``Tensor``: The ``torch.float32`` mask of shape ``[H, W]``, with
        the values ``0.0`` and ``1.0``.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> mask = generate_perlin_noise((64, 64))
        >>> mask.shape, mask.dtype
        (torch.Size([64, 64]), torch.float32)
        >>> set(mask.unique().tolist()) <= {0.0, 1.0}
        True

    """
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
    r"""Blend a texture into an image inside a random Perlin noise mask.

    The function draws a mask :math:`M` with `generate_perlin_noise`.
    Then it blends the image :math:`I` and the texture :math:`A`:

    .. math::

        I' = (1 - M) \odot I + (1 - \beta) M \odot A + \beta M \odot I

    The image does not change outside the mask. Inside the mask,
    ``beta`` is the weight of the image and ``1 - beta`` is the weight
    of the texture. ``H`` and ``W`` must be multiples of ``32``, see
    `generate_perlin_noise`.

    Args:
        img (``Tensor``): The clean image of shape ``[C, H, W]``.
        anomaly_img (``Tensor``): The texture image of shape
            ``[C, H, W]``.
        beta (float | None): The weight of the image inside the mask.
            ``None`` draws a value from ``[0, 0.8)``.

    Returns:
        ``tuple[Tensor, Tensor]``: The image with the anomaly, of shape
        ``[C, H, W]``, and the mask of shape ``[H, W]``. The mask is
        ``1.0`` inside the anomaly and ``0.0`` outside.

    Example:
        With ``beta=0.0``, the texture replaces the image inside the
        mask:

        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> img = torch.zeros(3, 64, 64)
        >>> texture = torch.ones(3, 64, 64)
        >>> augmented, mask = apply_anomaly_to_img(img, texture, beta=0.0)
        >>> augmented.shape, mask.shape
        (torch.Size([3, 64, 64]), torch.Size([64, 64]))
        >>> torch.equal(augmented, mask.expand(3, -1, -1))
        True

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
