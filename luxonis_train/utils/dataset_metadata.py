from luxonis_train.loaders import BaseLoaderTorch
from luxonis_train.utils import anchors_from_dataset


class DatasetMetadata:
    """Metadata about the dataset."""

    def __init__(
        self,
        *,
        classes: dict[str, list[str]] | None = None,
        n_keypoints: dict[str, int] | None = None,
        loader: BaseLoaderTorch | None = None,
    ):
        """An object containing metadata about the dataset. Used to infer the number of
        classes, number of keypoints, I{etc.} instead of passing them as arguments to
        the model.

        @type classes: dict[str, list[str]] | None
        @param classes: Dictionary mapping tasks to lists of class names.
        @type n_keypoints: dict[str, int] | None
        @param n_keypoints: Dictionary mapping tasks to the number of keypoints.
        @type loader: DataLoader | None
        @param loader: Dataset loader.
        """
        self._classes = classes or {}
        self._n_keypoints = n_keypoints or {}
        self._loader = loader

    @property
    def classes(self) -> dict[str, list[str]]:
        """Dictionary mapping label types to lists of class names.

        @type: dict[str, list[str]]
        @raises ValueError: If classes were not provided during initialization.
        """
        if self._classes is None:
            raise ValueError(
                "Trying to access `classes`, byt they were not"
                "provided during initialization."
            )
        return self._classes

    def n_classes(self, task: str | None) -> int:
        """Gets the number of classes for the specified task.

        @type task: str | None
        @param task: Task to get the number of classes for.
        @rtype: int
        @return: Number of classes for the specified label type.
        @raises ValueError: If the dataset loader was not provided during
            initialization.
        @raises ValueError: If the dataset contains different number of classes for
            different label types.
        """
        if task is not None:
            if task not in self.classes:
                raise ValueError(f"Task '{task}' is not present in the dataset.")
            return len(self.classes[task])
        n_classes = len(list(self.classes.values())[0])
        for classes in self.classes.values():
            if len(classes) != n_classes:
                raise ValueError(
                    "The dataset contains different number of classes for different tasks."
                )
        return n_classes

    def n_keypoints(self, task: str | None) -> int:
        """Gets the number of keypoints for the specified task.

        @type task: str | None
        @param task: Task to get the number of keypoints for.
        @rtype: int
        @return: Number of keypoints for the specified label type.
        @raises ValueError: If the dataset loader was not provided during initialization
            or if the dataset does not contain the specified task.
        """
        if task is not None:
            if task not in self._n_keypoints:
                raise ValueError(f"Task '{task}' is not present in the dataset.")
            return self._n_keypoints[task]
        if len(self._n_keypoints) > 1:
            raise ValueError(
                "The dataset specifies multiple keypoint tasks, "
                "please specify the 'task' argument to get the number of keypoints."
            )
        return next(iter(self._n_keypoints.values()))

    def class_names(self, task: str | None) -> list[str]:
        """Gets the class names for the specified task.

        @type task: str | None
        @param task: Task to get the class names for.
        @rtype: list[str]
        @return: List of class names for the specified label type.
        @raises ValueError: If the dataset loader was not provided during
            initialization.
        @raises ValueError: If the dataset contains different class names for different
            label types.
        """
        if task is not None:
            if task not in self.classes:
                raise ValueError(f"Task type {task} is not present in the dataset.")
            return self.classes[task]
        class_names = list(self.classes.values())[0]
        for classes in self.classes.values():
            if classes != class_names:
                raise ValueError(
                    "The dataset contains different class names for different tasks."
                )
        return class_names

    def autogenerate_anchors(self, num_heads: int) -> tuple[list[list[float]], float]:
        """Automatically generates anchors for the provided dataset.

        @type num_heads: int
        @param num_heads: Number of heads to generate anchors for.
        @rtype: tuple[list[list[float]], float]
        @return: List of anchors in [-1,6] format and recall of the anchors.
        @raises ValueError: If the dataset loader was not provided during
            initialization.
        """
        if self.loader is None:
            raise ValueError(
                "Cannot generate anchors without a dataset loader. "
                "Please provide a dataset loader to the constructor "
                "or call `set_loader` method."
            )

        proposed_anchors, recall = anchors_from_dataset(
            self.loader, n_anchors=num_heads * 3
        )
        return proposed_anchors.reshape(-1, 6).tolist(), recall

    def set_loader(self, loader: BaseLoaderTorch) -> None:
        """Sets the dataset loader.

        @type loader: DataLoader
        @param loader: Dataset loader.
        """
        self.loader = loader

    @classmethod
    def from_loader(cls, loader: BaseLoaderTorch) -> "DatasetMetadata":
        """Creates a L{DatasetMetadata} object from a L{LuxonisDataset}.

        @type dataset: LuxonisDataset
        @param dataset: Dataset to create the metadata from.
        @rtype: DatasetMetadata
        @return: Instance of L{DatasetMetadata} created from the provided dataset.
        """
        classes = loader.get_classes()
        n_keypoints = loader.get_n_keypoints()

        instance = cls(classes=classes, n_keypoints=n_keypoints)
        instance.set_loader(loader)
        return instance
