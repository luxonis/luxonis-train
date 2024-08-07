from typing import Annotated, Any, Literal, TypeVar

from luxonis_ml.data import LabelType
from pydantic import BaseModel, Field, ValidationError
from torch import Size, Tensor

Kwargs = dict[str, Any]
OutputTypes = Literal["boundingbox", "class", "keypoints", "segmentation", "features"]
Labels = dict[str, tuple[Tensor, LabelType]]

AttachIndexType = Literal["all"] | int | tuple[int, int] | tuple[int, int, int]
"""AttachIndexType is used to specify to which output of the prevoius node does the
current node attach to.

It can be either "all" (all outputs), an index of the output or a tuple of indices of
the output (specifying a range of outputs).
"""

T = TypeVar("T", Tensor, Size)
Packet = dict[str, list[T]]
"""Packet is a dictionary containing a list of objects of type T.

It is used to pass data between different nodes of the network graph.
"""


class IncompatibleException(Exception):
    """Raised when two parts of the model are incompatible with each other."""

    @classmethod
    def from_validation_error(cls, val_error: ValidationError, class_name: str):
        return cls(
            f"{class_name} received an input not conforming to the protocol. "
            f"Validation error: {val_error.errors(include_input=False, include_url=False)}."
        )

    @classmethod
    def from_missing_task(cls, task: str, present_tasks: list[str], class_name: str):
        return cls(
            f"{class_name} requires '{task}' label, but it was not found in "
            f"the label dictionary. Available labels: {present_tasks}."
        )


class BaseProtocol(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_task(cls) -> str:
        if len(cls.__annotations__) == 1:
            return list(cls.__annotations__)[0]
        raise ValueError(
            "Protocol must have exactly one field for automatic task inference. "
            "Implement custom `prepare` method in your attached module."
        )


class FeaturesProtocol(BaseProtocol):
    features: Annotated[list[Tensor], Field(min_length=1)]
