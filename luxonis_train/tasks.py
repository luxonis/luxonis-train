from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from types import UnionType
from typing import Any, get_args

from luxonis_ml.data import Category

__all__ = ["Metadata", "Task", "Tasks"]


class staticproperty:
    """Descriptor that exposes a zero-argument callable as a static property."""

    def __init__(self, func: Callable) -> None:
        """Create a static property descriptor.

        Args:
            func (Callable): Zero-argument callable used to compute the value.
        """
        self.func = func

    def __get__(self, *_) -> Any:
        """Evaluate the wrapped callable.

        Args:
            *_ (Any): Descriptor protocol arguments, ignored by this
                implementation.

        Returns:
            Any: Value returned by the wrapped callable.
        """
        return self.func()


@dataclass
class Metadata:
    """Metadata label requirement for a task.

    Attributes:
        name (str): Metadata label name.
        typ (UnionType | type): Accepted metadata value type or union of
            accepted types.
    """

    name: str
    typ: UnionType | type

    def __str__(self) -> str:
        """Return the metadata label path.

        Returns:
            str: Metadata label path.
        """
        return f"metadata/{self.name}"

    def __repr__(self) -> str:
        """Return the metadata label representation.

        Returns:
            str: Metadata label representation.
        """
        return str(self)

    def __hash__(self) -> int:
        """Return the hash of the metadata label path.

        Returns:
            int: Hash value.
        """
        return hash(str(self))

    def check_type(self, typ: UnionType | type) -> bool:
        """Check whether a type is accepted by this metadata requirement.

        Args:
            typ (UnionType | type): Type to check.

        Returns:
            bool: Whether `typ` is accepted.
        """
        if isinstance(self.typ, UnionType):
            return typ in get_args(self.typ)
        return typ == self.typ


@dataclass(frozen=True, unsafe_hash=True)
class Task(ABC):
    """Base task definition.

    Attributes:
        name (str): Task name used in model outputs and labels.
    """

    name: str

    @cached_property
    @abstractmethod
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        ...

    @property
    def main_output(self) -> str:
        """str: Main output name for this task."""
        return self.name


class Classification(Task):
    """Classification task definition."""

    def __init__(self):
        """Create a classification task."""
        super().__init__("classification")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return {"classification"}


class Segmentation(Task):
    """Semantic segmentation task definition."""

    def __init__(self):
        """Create a semantic segmentation task."""
        super().__init__("segmentation")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return {"segmentation"}


class InstanceBaseTask(Task):
    """Base task definition for instance-level tasks."""

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return {"boundingbox"}


class BoundingBox(InstanceBaseTask):
    """Bounding box detection task definition."""

    def __init__(self):
        """Create a bounding box detection task."""
        super().__init__("boundingbox")


class InstanceSegmentation(InstanceBaseTask):
    """Instance segmentation task definition."""

    def __init__(self):
        """Create an instance segmentation task."""
        super().__init__("instance_segmentation")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return super().required_labels | {"instance_segmentation"}


class InstanceKeypoints(InstanceBaseTask):
    """Instance keypoint task definition."""

    def __init__(self):
        """Create an instance keypoint task."""
        super().__init__("keypoints")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return super().required_labels | {"keypoints"}


class InstanceSegmentationKeypoints(InstanceBaseTask):
    """Instance segmentation and keypoint task definition."""

    def __init__(self):
        """Create an instance segmentation and keypoint task."""
        super().__init__("instance_segmentation_keypoints")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return super().required_labels | {"instance_segmentation", "keypoints"}


class Keypoints(Task):
    """Pointcloud keypoint task definition."""

    def __init__(self):
        """Create a pointcloud keypoint task."""
        super().__init__("pointcloud")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return {"keypoints"}


class Fomo(InstanceBaseTask):
    """FOMO detection task definition."""

    def __init__(self):
        """Create a FOMO detection task."""
        super().__init__("fomo")

    @property
    def main_output(self) -> str:
        """str: Main output name for this task."""
        return "heatmap"


class Embeddings(Task):
    """Embeddings task definition."""

    def __init__(self):
        """Create an embeddings task."""
        super().__init__("embeddings")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return {Metadata("id", int | Category)}


class AnomalyDetection(Task):
    """Anomaly detection task definition."""

    def __init__(self):
        """Create an anomaly detection task."""
        super().__init__("anomaly_detection")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return {"segmentation", "original_segmentation"}

    @property
    def main_output(self) -> str:
        """str: Main output name for this task."""
        return "segmentation"


class Ocr(Task):
    """OCR task definition."""

    def __init__(self):
        """Create an OCR task."""
        super().__init__("ocr")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        """set[str | Metadata]: Labels required by this task."""
        return {Metadata("text", str)}


class Tasks:
    """Factory namespace for task definitions."""

    @staticproperty
    def CLASSIFICATION() -> Classification:
        """Classification: Classification task definition."""
        return Classification()

    @staticproperty
    def SEGMENTATION() -> Segmentation:
        """Segmentation: Semantic segmentation task definition."""
        return Segmentation()

    @staticproperty
    def INSTANCE_SEGMENTATION() -> InstanceSegmentation:
        """InstanceSegmentation: Instance segmentation task definition."""
        return InstanceSegmentation()

    @staticproperty
    def BOUNDINGBOX() -> BoundingBox:
        """BoundingBox: Bounding box detection task definition."""
        return BoundingBox()

    @staticproperty
    def INSTANCE_KEYPOINTS() -> InstanceKeypoints:
        """InstanceKeypoints: Instance keypoint task definition."""
        return InstanceKeypoints()

    @staticproperty
    def KEYPOINTS() -> Keypoints:
        """Keypoints: Pointcloud keypoint task definition."""
        return Keypoints()

    @staticproperty
    def EMBEDDINGS() -> Embeddings:
        """Embeddings: Embeddings task definition."""
        return Embeddings()

    @staticproperty
    def ANOMALY_DETECTION() -> AnomalyDetection:
        """AnomalyDetection: Anomaly detection task definition."""
        return AnomalyDetection()

    @staticproperty
    def OCR() -> Ocr:
        """Ocr: OCR task definition."""
        return Ocr()

    @staticproperty
    def INSTANCE_SEGMENTATION_KEYPOINTS() -> InstanceSegmentationKeypoints:
        """InstanceSegmentationKeypoints: Instance segmentation and keypoint task definition."""
        return InstanceSegmentationKeypoints()

    @staticproperty
    def FOMO() -> Fomo:
        """Fomo: FOMO detection task definition."""
        return Fomo()
