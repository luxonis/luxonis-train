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


@dataclass
class Metadata:
    name: str

    @property
    def value(self):
        return f"metadata/{self.name}"

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.name)


Task: TypeAlias = TaskType | Metadata
# class TaskType:
#
#     CLASSIFICATION = SimpleTask.CLASSIFICATION
#     SEGMENTATION = SimpleTask.SEGMENTATION
#     INSTANCE_SEGMENTATION = SimpleTask.INSTANCE_SEGMENTATION
#     BOUNDINGBOX = SimpleTask.BOUNDINGBOX
#     KEYPOINTS = SimpleTask.KEYPOINTS
#     ARRAY = SimpleTask.ARRAY
