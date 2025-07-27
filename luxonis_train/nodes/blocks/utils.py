from collections.abc import Iterable
from typing import Protocol, TypeVar

from torch import Tensor, nn

T = TypeVar("T", int, tuple[int, ...])


class ModuleFactory(Protocol):
    def __call__(self, in_channels: int, out_channels: int) -> nn.Module: ...


def autopad(kernel_size: T, padding: T | None = None) -> T:
    """Compute padding based on kernel size.

    @type kernel_size: int | tuple[int, ...]
    @param kernel_size: Kernel size.
    @type padding: int | tuple[int, ...] | None
    @param padding: Padding. Defaults to None.

    @rtype: int | tuple[int, ...]
    @return: Computed padding. The output type is the same as the type of the
        C{kernel_size}.
    """
    if padding is not None:
        return padding
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(x // 2 for x in kernel_size)


def forward_gather(x: Tensor, modules: Iterable[nn.Module]) -> list[Tensor]:
    """Sequential forward pass through a list of modules, gathering
    intermediate outputs.

    @type x: Tensor
    @param x: Input tensor.
    @type modules: Iterable[nn.Module]
    @param modules: List of modules to apply.
    @rtype: list[Tensor]
    @return: List of intermediate outputs.
    """
    out = []
    for module in modules:
        x = module(x)
        out.append(x)
    return out
