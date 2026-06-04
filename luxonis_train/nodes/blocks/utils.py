from collections.abc import Iterable
from typing import Protocol, TypeVar

from torch import Tensor, nn

T = TypeVar("T", int, tuple[int, ...])


class ModuleFactory(Protocol):
    def __call__(self, in_channels: int, out_channels: int) -> nn.Module: ...


def autopad(kernel_size: T, padding: T | None = None) -> T:
    """Compute padding based on kernel size.

    Args:
        kernel_size (int | tuple[int, ...]): Kernel size.
        padding (int | tuple[int, ...] | None): Padding. Defaults to None.

    Returns:
        int | tuple[int, ...]: Computed padding. The output type is the same as the type of the ``kernel_size``.

    """
    if padding is not None:
        return padding
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(x // 2 for x in kernel_size)


def forward_gather(x: Tensor, modules: Iterable[nn.Module]) -> list[Tensor]:
    """Run modules sequentially and gather intermediate outputs.

    Args:
        x (Tensor): Input tensor.
        modules (Iterable[nn.Module]): List of modules to apply.

    Returns:
        list[Tensor]: List of intermediate outputs.

    """
    out = []
    for module in modules:
        x = module(x)
        out.append(x)
    return out
