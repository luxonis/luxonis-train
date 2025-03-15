from typing import Set

from bidict import bidict

from luxonis_train.loaders import BaseLoaderTorch


class DatasetMetadata:
    """Metadata about the dataset."""

    def __init__(
        self,
        *,
        classes: dict[str, dict[str, int]] | None = None,
        n_keypoints: dict[str, int] | None = None,
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
        self._loader = loader

    @property
    def task_names(self) -> Set[str]:
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
        n_classes = len(list(self._classes.values())[0])
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

    @classmethod
    def from_loader(cls, loader: BaseLoaderTorch) -> "DatasetMetadata":
        """Creates a L{DatasetMetadata} object from a L{LuxonisDataset}.

        @type dataset: LuxonisDataset
        @param dataset: Dataset to create the metadata from.
        @rtype: DatasetMetadata
        @return: Instance of L{DatasetMetadata} created from the
            provided dataset.
        """
        classes = loader.get_classes()
        n_keypoints = loader.get_n_keypoints()

        instance = cls(classes=classes, n_keypoints=n_keypoints, loader=loader)
        return instance
