"""The tasks a head can solve, and the labels each one requires.

A task connects a head to the losses, metrics, and visualizers that
attach to it. It gives the key of the main prediction of the head and
the labels that the loader must supply. `Tasks` gives an instance of
each built-in task.

"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from types import UnionType
from typing import Any, get_args

from luxonis_ml.data import Category

__all__ = ["Metadata", "Task", "Tasks"]


class staticproperty:
    """Descriptor that calls a function on each attribute access.

    `Tasks` uses it so that every attribute access returns a fresh task.

    """

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __get__(self, *_) -> Any:
        return self._func()


@dataclass
class Metadata:
    """A metadata label that a task requires.

    The string form of the label is ``"metadata/<name>"``. The key of
    the label in the loader output is ``"<task_name>/metadata/<name>"``,
    where ``<task_name>`` is the dataset task of the node. Two labels
    are equal when ``name`` and ``typ`` are equal. The hash depends only
    on ``name``.

    Attributes:
        name (str): The name of the metadata label, for example
            ``"id"``. The ``metadata_task_override`` field of a node
            config can rename it.
        typ (types.UnionType | type): The type that the label values
            must have, or a union of the accepted types.

    Example:
        >>> from luxonis_train.tasks import Metadata
        >>> label = Metadata("text", str)
        >>> str(label)
        'metadata/text'

    """

    name: str
    typ: UnionType | type

    def __str__(self) -> str:
        return f"metadata/{self.name}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def check_type(self, typ: UnionType | type) -> bool:
        """Check if the label accepts values of a type.

        When the ``typ`` attribute is a union, the method checks that
        the ``typ`` argument is one of the members of the union.
        Otherwise, it checks that the two types are equal. A subclass of
        an accepted type does not match.

        Args:
            typ (types.UnionType | type): The type to check, for example
                the type of the label in the dataset metadata.

        Returns:
            bool: ``True`` when the label accepts ``typ``.

        Example:
            >>> from luxonis_train.tasks import Metadata
            >>> label = Metadata("id", int | str)
            >>> label.check_type(str), label.check_type(float)
            (True, False)
            >>> Metadata("flag", int).check_type(bool)
            False

        """
        if isinstance(self.typ, UnionType):
            return typ in get_args(self.typ)
        return typ == self.typ


@dataclass(frozen=True, unsafe_hash=True)
class Task(ABC):
    """Base class for all tasks.

    A head sets its task in the ``task`` class attribute. A loss, a
    metric, or a visualizer lists the tasks it supports in
    ``supported_tasks``. Two tasks are equal when they have the same
    class and the same ``name``.

    A subclass must override `required_labels`. Python does not enforce
    this rule, because `functools.cached_property` hides the abstract
    method. An instance of a subclass without the override returns
    ``None`` for `required_labels`.

    Attributes:
        name (str): The name of the task. It is the default value of
            `main_output`.

    """

    name: str

    @cached_property
    @abstractmethod
    def required_labels(self) -> set[str | Metadata]:
        """The labels that the loader must supply for this task.

        An implementation returns a set of label types, such as
        ``"boundingbox"``, and `Metadata` labels. The key of a label in
        the loader output is ``"<task_name>/<label>"``. ``<task_name>``
        is the dataset task of the node, not the ``name`` of the task.
        The property computes the value once for each task instance.

        """
        ...

    @property
    def main_output(self) -> str:
        """The key of the main prediction of a head with this task.

        It is ``name`` unless a subclass overrides it. When ``forward``
        returns a tensor or a list of tensors, `BaseNode.run` puts the
        result under this key. An attached module fills some arguments
        from this key. The name of such an argument starts with ``pred``
        and has no underscore, for example ``predictions``.

        """
        return self.name


class Classification(Task):
    """The classification task.

    Its name and required label are both ``"classification"``.

    """

    def __init__(self):
        super().__init__("classification")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"classification"}


class Segmentation(Task):
    """The semantic segmentation task.

    Its name and required label are both ``"segmentation"``.

    """

    def __init__(self):
        super().__init__("segmentation")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"segmentation"}


class InstanceBaseTask(Task):
    """Base class for the tasks that detect object instances.

    The subclasses require the ``"boundingbox"`` label. Some subclasses
    add more labels.

    """

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"boundingbox"}


class BoundingBox(InstanceBaseTask):
    """The bounding box detection task.

    Its name is ``"boundingbox"``. It requires only the
    ``"boundingbox"`` label.

    """

    def __init__(self):
        super().__init__("boundingbox")


class InstanceSegmentation(InstanceBaseTask):
    """The instance segmentation task.

    Its name is ``"instance_segmentation"``. It requires bounding boxes
    and instance masks.

    """

    def __init__(self):
        super().__init__("instance_segmentation")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return super().required_labels | {"instance_segmentation"}


class InstanceKeypoints(InstanceBaseTask):
    """The keypoint detection task for object instances.

    Its name is ``"keypoints"``. It requires bounding boxes and
    keypoints.

    """

    def __init__(self):
        super().__init__("keypoints")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return super().required_labels | {"keypoints"}


class InstanceSegmentationKeypoints(InstanceBaseTask):
    """The instance segmentation and keypoint detection task.

    Its name is ``"instance_segmentation_keypoints"``. It requires
    bounding boxes, instance masks, and keypoints.

    """

    def __init__(self):
        super().__init__("instance_segmentation_keypoints")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return super().required_labels | {"instance_segmentation", "keypoints"}


class Keypoints(Task):
    """The keypoint task without bounding boxes.

    Its name is ``"pointcloud"``. Unlike `InstanceKeypoints`, it does
    not require the ``"boundingbox"`` label.

    """

    def __init__(self):
        super().__init__("pointcloud")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"keypoints"}


class Fomo(InstanceBaseTask):
    """The FOMO detection task.

    Its name is ``"fomo"``. A FOMO head predicts a heatmap of object
    centers. The task requires the ``"boundingbox"`` label, and its main
    prediction is ``"heatmap"``.

    """

    def __init__(self):
        super().__init__("fomo")

    @property
    def main_output(self) -> str:
        return "heatmap"


class Embeddings(Task):
    """The embedding task.

    Its name is ``"embeddings"``. It requires integer or categorical
    ``"metadata/id"`` labels.

    """

    def __init__(self):
        super().__init__("embeddings")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {Metadata("id", int | Category)}


class AnomalyDetection(Task):
    """The anomaly detection task.

    Its name is ``"anomaly_detection"``. `LuxonisLoaderPerlinNoise`
    supplies both of its labels. The ``"segmentation"`` label is the
    one-hot anomaly mask, of shape ``[2, H, W]``. The
    ``"original_segmentation"`` label is the input image without the
    added anomaly.

    """

    def __init__(self):
        super().__init__("anomaly_detection")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {"segmentation", "original_segmentation"}

    @property
    def main_output(self) -> str:
        return "segmentation"


class Ocr(Task):
    """The optical character recognition task.

    Its name is ``"ocr"``. It requires string ``"metadata/text"``
    labels.

    """

    def __init__(self):
        super().__init__("ocr")

    @cached_property
    def required_labels(self) -> set[str | Metadata]:
        return {Metadata("text", str)}


class Tasks:
    """The namespace that gives an instance of each built-in task.

    Each access to an attribute builds a new task instance. The new
    instances are equal to each other, so a test such as
    ``node.task in supported_tasks`` works.

    Example:
        >>> from luxonis_train.tasks import Tasks
        >>> task = Tasks.INSTANCE_KEYPOINTS
        >>> task.name, task.main_output
        ('keypoints', 'keypoints')
        >>> sorted(task.required_labels)
        ['boundingbox', 'keypoints']
        >>> Tasks.FOMO.main_output
        'heatmap'
        >>> Tasks.BOUNDINGBOX == Tasks.BOUNDINGBOX
        True

    """

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
    def INSTANCE_SEGMENTATION_KEYPOINTS() -> InstanceSegmentationKeypoints:
        return InstanceSegmentationKeypoints()

    @staticproperty
    def FOMO() -> Fomo:
        return Fomo()
