from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from types import UnionType
from typing import get_args

from luxonis_ml.data import Category


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


class Tasks:
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

    class InstanceSegmentation(Task):
        def __init__(self):
            super().__init__("instance_segmentation")

        @cached_property
        def required_labels(self) -> set[str | Metadata]:
            return {"instance_segmentation", "boundingbox"}

    class BoundingBox(Task):
        def __init__(self):
            super().__init__("boundingbox")

        @cached_property
        def required_labels(self) -> set[str | Metadata]:
            return {"boundingbox"}

    class Keypoints(Task):
        def __init__(self):
            super().__init__("keypoints")

        @cached_property
        def required_labels(self) -> set[str | Metadata]:
            return {"keypoints", "boundingbox"}

    class Pointcloud(Task):
        def __init__(self):
            super().__init__("pointcloud")

        @cached_property
        def required_labels(self) -> set[str | Metadata]:
            return {"keypoints"}

        @property
        def main_output(self) -> str:
            return "keypoints"

    class Embeddings(Task):
        def __init__(self):
            super().__init__("embeddings")

        @cached_property
        def required_labels(self) -> set[str | Metadata]:
            return {Metadata("id", int | Category)}

        @property
        def main_output(self) -> str:
            return "embeddings"

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
            return {Metadata("text", str), Metadata("text_length", int)}

    class Fomo(Task):
        def __init__(self):
            super().__init__("fomo")

        @cached_property
        def required_labels(self) -> set[str | Metadata]:
            return {"boundingbox"}

        @property
        def main_output(self) -> str:
            return "heatmap"

    CLASSIFICATION = Classification()
    SEGMENTATION = Segmentation()
    INSTANCE_SEGMENTATION = InstanceSegmentation()
    BOUNDINGBOX = BoundingBox()
    KEYPOINTS = Keypoints()
    POINTCLOUD = Pointcloud()
    EMBEDDINGS = Embeddings()
    ANOMALY_DETECTION = AnomalyDetection()
    OCR = Ocr()
    FOMO = Fomo()
