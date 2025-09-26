from collections.abc import Iterator
from pprint import pformat
from typing import Any

from bidict import bidict
from luxonis_ml.data import Category

from luxonis_train.loaders import BaseLoaderTorch


class DatasetMetadata:
    """Metadata about the dataset."""

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
        """An object containing metadata about the dataset. Used to
        infer the number of classes, number of keypoints, I{etc.}
        instead of passing them as arguments to the model.

        @type classes: dict[str, dict[str, int]] | None
        @param classes: Dictionary mapping tasks to the classes.
        @type n_keypoints: dict[str, int] | None
        @param n_keypoints: Dictionary mapping tasks to the number of
            keypoints.
        @type loader: DataLoader | None
        @param loader: Dataset loader.
        """
        self._classes = classes or {}
        self._n_keypoints = n_keypoints or {}
        self._metadata_types = metadata_types or {}
        self._loader = loader

    def __str__(self) -> str:
        return pformat(self.dump())

    def __repr__(self) -> str:
        return str(self)

    def __rich_repr__(self) -> Iterator[tuple[str, Any]]:
        yield from self.dump().items()

    def dump(self) -> dict[str, Any]:
        """Dumps the metadata to a dictionary.

        @rtype: dict[str, dict[str, int] | int | dict[str, type]]
        @return: Dictionary containing the metadata.
        """
        return {
            "classes": {k: dict(v) for k, v in self._classes.items()},
            "n_keypoints": dict(self._n_keypoints),
            "metadata_types": {
                k: v.__name__ for k, v in self._metadata_types.items()
            },
        }

    @property
    def task_names(self) -> set[str]:
        """Gets the names of the tasks present in the dataset.

        @rtype: set[str]
        @return: Names of the tasks present in the dataset.
        """
        return set(self._classes.keys())

    def n_classes(self, task_name: str | None = None) -> int:
        """Gets the number of classes for the specified task.

        @type task_name: str | None
        @param task_name: Task to get the number of classes for.
        @rtype: int
        @return: Number of classes for the specified task type.
        @raises ValueError: If the C{task} is not present in the
            dataset.
        @raises RuntimeError: If the C{task} was not provided and the
            dataset contains different number of classes for different
            task types.
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
        """Gets the number of keypoints for the specified task.

        @type task_name: str | None
        @param task_name: Task to get the number of keypoints for.
        @rtype: int
        @return: Number of keypoints for the specified task type.
        @raises ValueError: If the C{task} is not present in the
            dataset.
        @raises RuntimeError: If the C{task} was not provided and the
            dataset contains different number of keypoints for different
            task types.
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
        """Gets the class names for the specified task.

        @type task_name: str | None
        @param task_name: Task to get the class names for.
        @rtype: list[str]
        @return: List of class names for the specified task type.
        @raises ValueError: If the C{task} is not present in the
            dataset.
        @raises RuntimeError: If the C{task} was not provided and the
            dataset contains different class names for different label
            types.
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
        """Gets the types of metadata for the dataset.

        @rtype: dict[str, type[int] | type[Category] | type[float] |
            type[str]
        @return: Dictionary mapping metadata names to their types.
        """
        if self._metadata_types is None:
            raise RuntimeError("The dataset does define metadata types.")
        return self._metadata_types

    @classmethod
    def from_loader(cls, loader: BaseLoaderTorch) -> "DatasetMetadata":
        """Creates a L{DatasetMetadata} object from a L{LuxonisDataset}.

        @type loader: LuxonisDataset
        @param loader: Loader to read the metadata from.
        @rtype: DatasetMetadata
        @return: Instance of L{DatasetMetadata} created from the
            provided dataset.
        """
        return cls(
            classes=loader.get_classes(),
            n_keypoints=loader.get_n_keypoints(),
            metadata_types=loader.get_metadata_types(),
            loader=loader,
        )
