from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from types import UnionType
from typing import Any, get_args

from luxonis_ml.data import Category

__all__ = ["Metadata", "Task", "Tasks"]


class staticproperty:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __get__(self, *_) -> Any:
        return self.func()


@dataclass
class Metadata:
    name: str
    typ: UnionType | type

    def __str__(self) -> str:
        return f"metadata/{self.name}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def check_type(self, typ: UnionType | type) -> bool:
        if isinstance(self.typ, UnionType):
            return typ in get_args(self.typ)
        return typ == self.typ


@dataclass(frozen=True, unsafe_hash=True)
class Task(ABC):
    name: str

    @cached_property
    @abstractmethod
    def required_labels(self) -> set[str | Metadata]: ...

    @property
    def main_output(self) -> str:
        return self.name


class Classification(Task):
    def __init__(self):
        super().__init__("classification")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"classification"}


class Segmentation(Task):
    def __init__(self):
        super().__init__("segmentation")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"segmentation"}


class InstanceBaseTask(Task):
    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"boundingbox"}


class BoundingBox(InstanceBaseTask):
    def __init__(self):
        super().__init__("boundingbox")


class InstanceSegmentation(InstanceBaseTask):
    def __init__(self):
        super().__init__("instance_segmentation")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return super().required_labels | {"instance_segmentation"}


class InstanceKeypoints(InstanceBaseTask):
    def __init__(self):
        super().__init__("keypoints")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return super().required_labels | {"keypoints"}


class Keypoints(Task):
    def __init__(self):
        super().__init__("pointcloud")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"keypoints"}


class Fomo(InstanceBaseTask):
    def __init__(self):
        super().__init__("fomo")

    @property
    def main_output(self) -> str:
        return "heatmap"


class Embeddings(Task):
    def __init__(self):
        super().__init__("embeddings")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {Metadata("id", int | Category)}


class AnomalyDetection(Task):
    def __init__(self):
        super().__init__("anomaly_detection")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"segmentation", "original_segmentation"}

    @property
    def main_output(self) -> str:
        return "segmentation"


class Ocr(Task):
    def __init__(self):
        super().__init__("ocr")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {Metadata("text", str)}


class Tasks:
    @staticproperty
    def CLASSIFICATION() -> Classification:
        return Classification()

    @staticproperty
    def SEGMENTATION() -> Segmentation:
        return Segmentation()

    @staticproperty
    def INSTANCE_SEGMENTATION() -> InstanceSegmentation:
        return InstanceSegmentation()

    @staticproperty
    def BOUNDINGBOX() -> BoundingBox:
        return BoundingBox()

    @staticproperty
    def INSTANCE_KEYPOINTS() -> InstanceKeypoints:
        return InstanceKeypoints()

    @staticproperty
    def KEYPOINTS() -> Keypoints:
        return Keypoints()

    @staticproperty
    def EMBEDDINGS() -> Embeddings:
        return Embeddings()

    @staticproperty
    def ANOMALY_DETECTION() -> AnomalyDetection:
        return AnomalyDetection()

    @staticproperty
    def OCR() -> Ocr:
        return Ocr()

    @staticproperty
    def FOMO() -> Fomo:
        return Fomo()
