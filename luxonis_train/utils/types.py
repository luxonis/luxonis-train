from typing import Any, Literal, TypeVar

from luxonis_ml.data import LabelType
from torch import Size, Tensor

Kwargs = dict[str, Any]
"""Kwargs is a dictionary containing keyword arguments."""

Labels = dict[str, tuple[Tensor, LabelType]]
"""Labels is a dictionary containing a tuple of tensors and their
corresponding label type."""

AttachIndexType = Literal["all"] | int | tuple[int, int] | tuple[int, int, int]
"""AttachIndexType is used to specify to which output of the prevoius
node does the current node attach to.

It can be either "all" (all outputs), an index of the output or a tuple
of indices of the output (specifying a range of outputs).
"""

T = TypeVar("T", Tensor, Size)
Packet = dict[str, list[T]]
"""Packet is a dictionary containing a list of objects of type T.

It is used to pass data between different nodes of the network graph.
"""
