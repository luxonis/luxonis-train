from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    BOUNDINGBOX = "boundingbox"
    KEYPOINTS = "keypoints"
    ARRAY = "array"


@dataclass(unsafe_hash=True)
class Metadata:
    name: str

    @property
    def value(self):
        return f"metadata/{self.name}"

    def __str__(self) -> str:
        return self.value


Task: TypeAlias = TaskType | Metadata
