"""The type aliases shared across the package."""

from typing import Literal, TypeAlias, TypeVar

from torch import Size, Tensor

View: TypeAlias = Literal["train", "val", "test"]
"""The name of a dataset view: ``"train"``, ``"val"``, or ``"test"``.

`LuxonisModel` builds one loader for each view.

"""

Labels: TypeAlias = dict[str, Tensor]
"""The label tensors of a sample or a batch, keyed by task and label.

A key has the form ``"<task_name>/<label>"``, where ``<task_name>`` is
the dataset task. Examples are ``"detection/boundingbox"`` and
``"detection/metadata/id"``.

"""

AttachIndexType: TypeAlias = (
    Literal["all"] | int | tuple[int, int] | tuple[int, int, int] | None
)
"""The outputs of the input node that a node reads.

The value is ``"all"`` for every output, or an integer index for one
output. A tuple of two integers selects a range, and a third integer
sets the step of the range. ``None`` means that the index is not set.
`BaseNode.get_attached` applies the index.

"""

T = TypeVar("T", Tensor, Size)
"""The value type of a `Packet`: ``Tensor`` or ``Size``."""

Packet: TypeAlias = dict[str, list[T] | T]
"""A dictionary that maps output names to values or to lists of values.

The values of one packet are all tensors or all sizes. A node reads
packets of tensors and returns a packet of tensors, for example
``{"features": [f1, f2]}``. A packet of sizes holds the shapes of the
outputs of a node.

"""
