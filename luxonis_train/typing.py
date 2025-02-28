from typing import Any, Iterable, Literal, TypeGuard, TypeVar

import typeguard
from torch import Size, Tensor

Labels = dict[str, Tensor]
"""Labels is a dictionary mapping task names to tensors."""

AttachIndexType = (
    Literal["all"] | int | tuple[int, int] | tuple[int, int, int] | None
)
"""AttachIndexType is used to specify to which output of the prevoius
node does the current node attach to.

It can be either "all" (all outputs), an index of the output or a tuple
of indices of the output (specifying a range of outputs).
"""

T = TypeVar("T", Tensor, Size)
Packet = dict[str, list[T] | T]
"""Packet is a dictionary containing a list of objects of type T.

It is used to pass data between different nodes of the network graph.
"""

K = TypeVar("K")


def check_type(value: Any, type_: type[K]) -> TypeGuard[K]:
    """Checks if the value has the correct type.

    Args:
        value: The value to check.
        type_: The expected type of the value.

    Returns:
        The value if the type is correct.

    Raises:
        TypeError: If the type is incorrect.
    """
    try:
        typeguard.check_type(value, type_)
    except typeguard.TypeCheckError:
        return False
    return True


def all_not_none(values: Iterable[Any]) -> bool:
    """Checks if none of the values in the iterable is C{None}"""
    return all(v is not None for v in values)


def any_not_none(values: Iterable[Any]) -> bool:
    """Checks if at least one value in the iterable is not C{None}"""
    return any(v is not None for v in values)
