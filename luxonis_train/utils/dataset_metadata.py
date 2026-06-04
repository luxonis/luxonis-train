from collections.abc import Iterator
from pprint import pformat
from typing import Any

from bidict import bidict
from luxonis_ml.data import Category

from luxonis_train.loaders import BaseLoaderTorch


class DatasetMetadata:
    """Dataset metadata used to configure model outputs.

    The metadata includes class definitions, keypoint counts, and
    additional label metadata types.

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
        """Create dataset metadata.

        Args:
            classes (dict[str, dict[str, int]] | None): Dictionary mapping
                task names to class-name-to-index mappings.
            n_keypoints (dict[str, int] | None): Dictionary mapping task names
                to the number of keypoints.
            metadata_types (dict[str, type[int] | type[Category] | type[float] | type[str]] | None):
                Dictionary mapping metadata names to metadata value types.
            loader (BaseLoaderTorch | None): Dataset loader associated with
                this metadata.

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
        """Dump the metadata to a dictionary.

        Returns:
            dict[str, Any]: Dictionary containing the metadata.

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
        """Set[str]: Names of the tasks present in the dataset."""
        return set(self._classes.keys())

    def n_classes(self, task_name: str | None = None) -> int:
        """Get the number of classes for the specified task.

        Args:
            task_name (str | None): Task to get the number of classes for.
                Defaults to ``None``.

        Returns:
            int: Number of classes for the specified task.

        Raises:
            ValueError: If `task_name` is not present in the dataset.
            RuntimeError: If `task_name` was not provided and the dataset
                contains different numbers of classes for different tasks.

        """
        if task_name is not None:
            if task_name not in self._classes:
                raise ValueError(
                    f"Task '{task_name}' is not present in the dataset."
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
        """Get the number of keypoints for the specified task.

        Args:
            task_name (str | None): Task to get the number of keypoints for.
                Defaults to ``None``.

        Returns:
            int: Number of keypoints for the specified task.

        Raises:
            ValueError: If `task_name` is not present in the dataset.
            RuntimeError: If `task_name` was not provided and the dataset
                contains different numbers of keypoints for different tasks.

        """
        if task_name is not None:
            if task_name not in self._n_keypoints:
                raise ValueError(
                    f"Task '{task_name}' is not present in the dataset."
                )
            return self._n_keypoints[task_name]
        n_keypoints = next(iter(self._n_keypoints.values()))
        for n in self._n_keypoints.values():
            if n != n_keypoints:
                raise RuntimeError(
                    "The dataset contains different number of keypoints for different tasks. "
                    "Please specify the 'task' argument to get the number of keypoints."
                )
        return n_keypoints

    def classes(self, task_name: str | None = None) -> bidict[str, int]:
        """Get the class names for the specified task.

        Args:
            task_name (str | None): Task to get the class names for. Defaults
                to ``None``.

        Returns:
            bidict[str, int]: Bidirectional dictionary mapping class names to
            their indices for the specified task.

        Raises:
            ValueError: If `task_name` is not present in the dataset.
            RuntimeError: If `task_name` was not provided and the dataset
                contains different class names for different tasks.

        """
        if task_name is not None:
            if task_name not in self._classes:
                raise ValueError(
                    f"Task '{task_name}' is not present in the dataset."
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
        """Dict[str, type[int] | type[Category] | type[float] |
        type[str]]: Metadata names mapped to their types.

        Raises:
            RuntimeError: If the dataset does not define metadata types.

        """
        if self._metadata_types is None:
            raise RuntimeError("The dataset does define metadata types.")
        return self._metadata_types

    @classmethod
    def from_loader(cls, loader: BaseLoaderTorch) -> "DatasetMetadata":
        """Create a `DatasetMetadata` object from a dataset loader.

        Args:
            loader (BaseLoaderTorch): Loader to read the metadata from.

        Returns:
            DatasetMetadata: Instance created from the provided loader.

        """
        return cls(
            classes=loader.get_classes(),
            n_keypoints=loader.get_n_keypoints(),
            metadata_types=loader.get_metadata_types(),
            loader=loader,
        )
