"""The dataset metadata that the nodes read.

The metadata holds the class names, the keypoint counts, and the
metadata label types that come from the loader.

"""

from collections.abc import Iterator
from pprint import pformat
from typing import Any

from bidict import bidict
from luxonis_ml.data import Category

from luxonis_train.loaders import BaseLoaderTorch


class DatasetMetadata:
    """The class names, keypoint counts, and metadata types of a dataset.

    A node reads its class count, class names, and keypoint count from
    this object, so the config does not have to set them. `from_loader`
    creates the object from a loader. `dump` returns a dictionary that
    a checkpoint stores and that the constructor accepts back as
    keyword arguments.

    Example:
        >>> metadata = DatasetMetadata(
        ...     classes={"detection": {"car": 0, "person": 1}},
        ...     n_keypoints={"detection": 17},
        ... )
        >>> metadata.n_classes("detection")
        2
        >>> metadata.classes().inverse[0]
        'car'
        >>> metadata.n_keypoints("segmentation")
        0

    """

    def __init__(
        self,
        *,
        classes: dict[str, dict[str, int]] | None = None,
        n_keypoints: dict[str, int] | None = None,
        metadata_types: dict[
            str, type[int] | type[Category] | type[float] | type[str]
        ]
        | None = None,
        loader: BaseLoaderTorch | None = None,
    ):
        """Initialize the metadata from plain dictionaries.

        A value in ``metadata_types`` can also be a type name: one of
        ``"int"``, ``"float"``, ``"str"``, or ``"Category"``. The
        constructor converts the name to the type. This is the form
        `dump` writes and a checkpoint stores.

        Args:
            classes (dict[str, dict[str, int]] | None): Task names mapped to
                the class names of the task and their indices. ``None``
                means no tasks.
            n_keypoints (dict[str, int] | None): Task names mapped to the
                number of keypoints of the task. ``None`` means no
                keypoints.
            metadata_types (``dict[str, type[int] | type[Category] | type[float] | type[str]] | None``):
                Metadata label names, such as
                ``"<task>/metadata/<name>"``, mapped to the type of their
                values. ``None`` means no metadata labels.
            loader (BaseLoaderTorch | None): The loader that gave the
                metadata. The object keeps a reference to it and does not
                use it.

        Raises:
            ValueError: When a type name in ``metadata_types`` is not one
                of the four supported names.

        """
        self._classes = classes or {}
        self._n_keypoints = n_keypoints or {}
        metadata_types = metadata_types or {}
        self._metadata_types = {
            k: self._parse_type(v) if isinstance(v, str) else v
            for k, v in metadata_types.items()
        }
        self._loader = loader

    def __str__(self) -> str:
        return pformat(self.dump())

    def __repr__(self) -> str:
        return str(self)

    def __rich_repr__(self) -> Iterator[tuple[str, Any]]:
        yield from self.dump().items()

    def dump(self) -> dict[str, Any]:
        """Dump the metadata to a dictionary of plain values.

        The constructor accepts the result back as keyword arguments.
        This is how a checkpoint stores and restores the metadata.

        Returns:
            ``dict[str, Any]``: A dictionary with the keys ``"classes"``,
            ``"n_keypoints"``, and ``"metadata_types"``. The metadata
            types appear as type names, for example ``"str"``.

        Example:
            >>> metadata = DatasetMetadata(
            ...     classes={"detection": {"car": 0}},
            ...     metadata_types={"color": str},
            ... )
            >>> metadata.dump()
            {'classes': {'detection': {'car': 0}},
             'n_keypoints': {},
             'metadata_types': {'color': 'str'}}
            >>> DatasetMetadata(**metadata.dump()).metadata_types
            {'color': <class 'str'>}

        """
        return {
            "classes": {k: dict(v) for k, v in self._classes.items()},
            "n_keypoints": dict(self._n_keypoints),
            "metadata_types": {
                k: v.__name__ for k, v in self._metadata_types.items()
            },
        }

    @staticmethod
    def _parse_type(type_name: str) -> type:
        if type_name == "int":
            return int
        if type_name == "float":
            return float
        if type_name == "str":
            return str
        if type_name == "Category":
            return Category
        raise ValueError(f"Unknown type name: {type_name}")

    @property
    def task_names(self) -> set[str]:
        """The names of all tasks in the class mapping.

        A task with an empty class mapping is also in the set.

        """
        return set(self._classes.keys())

    def n_classes(self, task_name: str | None = None) -> int:
        """Get the number of classes of a task.

        Args:
            task_name (str | None): The task to read. ``None`` means all
                tasks, which must then have the same number of classes.

        Returns:
            int: The number of classes of the task.

        Raises:
            ValueError: When ``task_name`` is not a task of the dataset.
            RuntimeError: When ``task_name`` is ``None`` and the tasks
                have different numbers of classes.
            StopIteration: When ``task_name`` is ``None`` and the
                metadata has no tasks.

        """
        if task_name is not None:
            if task_name not in self._classes:
                raise ValueError(
                    f"Task '{task_name}' is not present in the dataset. "
                    f"Available tasks: {self.task_names}"
                )
            return len(self._classes[task_name])
        n_classes = len(next(iter(self._classes.values())))
        for classes in self._classes.values():
            if len(classes) != n_classes:
                raise RuntimeError(
                    "The dataset contains different number of classes for different tasks. "
                    "Please specify the 'task' argument to get the number of classes."
                )
        return n_classes

    def n_keypoints(self, task_name: str | None = None) -> int:
        """Get the number of keypoints of a task.

        Args:
            task_name (str | None): The task to read. ``None`` means all
                tasks, which must then have the same number of keypoints.

        Returns:
            int: The number of keypoints of the task. ``0`` when
            ``task_name`` has no keypoint count, for example a task that
            is not in the dataset.

        Raises:
            RuntimeError: When ``task_name`` is ``None`` and the tasks
                have different numbers of keypoints.
            StopIteration: When ``task_name`` is ``None`` and the
                metadata has no keypoint counts.

        """
        if task_name is not None:
            return self._n_keypoints.get(task_name, 0)
        n_keypoints = next(iter(self._n_keypoints.values()))
        for n in self._n_keypoints.values():
            if n != n_keypoints:
                raise RuntimeError(
                    "The dataset contains different number of keypoints for different tasks. "
                    "Please specify the 'task' argument to get the number of keypoints."
                )
        return n_keypoints

    def classes(self, task_name: str | None = None) -> bidict[str, int]:
        """Get the class names and indices of a task.

        Args:
            task_name (str | None): The task to read. ``None`` means all
                tasks, which must then have the same classes.

        Returns:
            ``bidict[str, int]``: A new bidirectional dictionary that maps
            the class names to the class indices. Its ``inverse`` maps
            the indices back to the names.

        Raises:
            ValueError: When ``task_name`` is not a task of the dataset.
            RuntimeError: When ``task_name`` is ``None`` and the tasks
                have different classes.
            StopIteration: When ``task_name`` is ``None`` and the
                metadata has no tasks.

        """
        if task_name is not None:
            if task_name not in self._classes:
                raise ValueError(
                    f"Task '{task_name}' is not present in the dataset. "
                    f"Available tasks: {self.task_names}"
                )
            return bidict(self._classes[task_name])
        classes = next(iter(self._classes.values()))
        for c in self._classes.values():
            if c != classes:
                raise RuntimeError(
                    "The dataset contains different class "
                    "definitions for different tasks."
                )
        return bidict(classes)

    @property
    def metadata_types(
        self,
    ) -> dict[str, type[int] | type[Category] | type[float] | type[str]]:
        """The metadata label names mapped to the type of their values.

        The dictionary is empty when the dataset has no metadata labels.

        """
        if self._metadata_types is None:
            raise RuntimeError("The dataset does define metadata types.")
        return self._metadata_types

    @classmethod
    def from_loader(cls, loader: BaseLoaderTorch) -> "DatasetMetadata":
        """Create the metadata from a loader.

        The method reads ``loader.get_classes()``,
        ``loader.get_n_keypoints()``, and ``loader.get_metadata_types()``.
        The new object keeps a reference to ``loader``.

        Args:
            loader (BaseLoaderTorch): The loader to read.

        Returns:
            DatasetMetadata: The metadata of the dataset of the loader.

        """
        return cls(
            classes=loader.get_classes(),
            n_keypoints=loader.get_n_keypoints(),
            metadata_types=loader.get_metadata_types(),
            loader=loader,
        )
