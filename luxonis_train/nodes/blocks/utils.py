"""Helpers for building blocks: the padding that keeps the size of a
convolution, the type of a block factory, and a forward pass that
collects intermediate outputs.
"""

from collections.abc import Iterable
from typing import Protocol, TypeVar

from torch import Tensor, nn

T = TypeVar("T", int, tuple[int, ...])


class ModuleFactory(Protocol):
    """Type of a callable that builds a block from its channel counts.

    A matching callable takes ``in_channels`` and ``out_channels`` and
    returns a `torch.nn.Module`. A block class such as
    `GeneralReparameterizableBlock` matches. `BottleRep` calls its
    factory with keyword arguments.

    """

    def __call__(self, in_channels: int, out_channels: int) -> nn.Module: ...


def autopad(kernel_size: T, padding: T | None = None) -> T:
    """Compute the padding that keeps the size of a convolution.

    The padding is half of the kernel size, rounded down. It keeps the
    size for an odd kernel size, a stride of ``1``, and a dilation of
    ``1``.

    Args:
        kernel_size (``int | tuple[int, ...]``): The kernel size, as one
            value or one value for each axis.
        padding (``int | tuple[int, ...] | None``): An explicit padding.
            ``None`` selects the computed padding.

    Returns:
        ``int | tuple[int, ...]``: ``padding`` when it is not ``None``.
        Otherwise ``kernel_size // 2`` for each value, with the type of
        ``kernel_size``.

    Example:
        >>> from luxonis_train.nodes.blocks import autopad
        >>> autopad(3)
        1
        >>> autopad((3, 5))
        (1, 2)
        >>> autopad(3, padding=0)
        0

    """
    if padding is not None:
        return padding
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(x // 2 for x in kernel_size)


def forward_gather(x: Tensor, modules: Iterable[nn.Module]) -> list[Tensor]:
    """Run modules in sequence and collect the output of each module.

    Each module takes the output of the module before it. The first
    module takes ``x``.

    Args:
        x (``Tensor``): The input of the first module.
        modules (``Iterable[nn.Module]``): The modules, in the order in
            which they run.

    Returns:
        ``list[Tensor]``: The output of each module, in the same order.
        The list does not hold ``x``.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from luxonis_train.nodes.blocks.utils import forward_gather
        >>> pools = [nn.MaxPool2d(2), nn.MaxPool2d(2)]
        >>> outputs = forward_gather(torch.zeros(1, 1, 8, 8), pools)
        >>> [tuple(output.shape) for output in outputs]
        [(1, 1, 4, 4), (1, 1, 2, 2)]

    """
    out = []
    for module in modules:
        x = module(x)
        out.append(x)
    return out
