from collections.abc import Iterable
from typing import Protocol, TypeVar

from torch import Tensor, nn

T = TypeVar("T", int, tuple[int, ...])


class ModuleFactory(Protocol):
    def __call__(self, in_channels: int, out_channels: int) -> nn.Module: ...


def autopad(kernel_size: T, padding: T | None = None) -> T:
    """Compute padding based on kernel size.

    Args:
        kernel_size: Kernel size.
        padding: Padding. Defaults to None.

    Returns:
        Computed padding. The output type is the same as the type of
        the ``kernel_size``.

    """
    if padding is not None:
        return padding
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(x // 2 for x in kernel_size)


def forward_gather(x: Tensor, modules: Iterable[nn.Module]) -> list[Tensor]:
    """Sequential forward pass through a list of modules, gathering
    intermediate outputs.

    Args:
        x: Input tensor.
        modules: List of modules to apply.

    Returns:
        List of intermediate outputs.

    """
    out = []
    for module in modules:
        x = module(x)
        out.append(x)
    return out
