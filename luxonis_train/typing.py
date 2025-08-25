import inspect
from collections.abc import Callable, Collection
from inspect import Parameter
from typing import Literal, TypeAlias, TypeVar

from torch import Size, Tensor

View: TypeAlias = Literal["train", "val", "test"]

Labels: TypeAlias = dict[str, Tensor]
"""Labels is a dictionary mapping task names to tensors."""

AttachIndexType: TypeAlias = (
    Literal["all"] | int | tuple[int, int] | tuple[int, int, int] | None
)
"""AttachIndexType is used to specify to which output of the prevoius
node does the current node attach to.

It can be either "all" (all outputs), an index of the output or a tuple
of indices of the output (specifying a range of outputs).
"""

T = TypeVar("T", Tensor, Size)
Packet: TypeAlias = dict[str, list[T] | T]
"""Packet is a dictionary containing either a single instance of a list
of either `torch.Tensor`s or `torch.Size`s.

Packets are used to pass data between nodes of the network graph.
"""


def get_signature(
    func: Callable, exclude: Collection[str] | None = None
) -> dict[str, Parameter]:
    """Get the signature of a function, excluding certain parameters
    like 'self' and 'kwargs'.

    @type func: Callable
    @param func: The function to get the signature of.
    @type exclude: Collection[str] | None
    @param exclude: A collection of parameter names to exclude from the
        signature. Defaults to None, which excludes 'self' and 'kwargs'.
    @rtype: dict[str, Parameter]
    @return: A dictionary mapping parameter names to their Parameter
        objects, excluding the specified parameters.
    """
    exclude = set(exclude or [])
    exclude |= {"self", "kwargs"}
    signature = dict(inspect.signature(func).parameters)
    return {
        name: param for name, param in signature.items() if name not in exclude
    }
