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
