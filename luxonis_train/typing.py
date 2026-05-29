from typing import Literal, TypeAlias, TypeVar

from torch import Size, Tensor

View: TypeAlias = Literal["train", "val", "test"]
"""TypeAlias: Dataset split name."""

Labels: TypeAlias = dict[str, Tensor]
"""TypeAlias: Dictionary mapping task names to tensors."""

AttachIndexType: TypeAlias = (
    Literal["all"] | int | tuple[int, int] | tuple[int, int, int] | None
)
"""TypeAlias: Output index specification for graph node attachment.

It can be ``"all"`` for all outputs, an integer output index, or a tuple of
output indices specifying a range of outputs.
"""

T = TypeVar("T", Tensor, Size)
Packet: TypeAlias = dict[str, list[T] | T]
"""TypeAlias: Dictionary containing tensors, sizes, or lists of them.

Packets are used to pass data between nodes of the network graph.
"""
