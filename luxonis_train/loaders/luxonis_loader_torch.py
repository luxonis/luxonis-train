"""The default loader, which reads a ``LuxonisDataset``.

`LuxonisLoaderTorch` opens a dataset that exists, or parses a directory
into a new dataset.

"""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from loguru import logger
from luxonis_ml.data import Category, LuxonisDataset, LuxonisLoader
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import Params
from torch import Size, Tensor
from typing_extensions import override

from luxonis_train.loaders import BaseLoaderTorch
from luxonis_train.typing import Labels


class LuxonisLoaderTorch(BaseLoaderTorch):
    """The default loader, which reads a ``LuxonisDataset``.

    The loader wraps a ``LuxonisLoader`` from ``luxonis_ml``. The
    ``LuxonisLoader`` reads the images and the labels of the splits in
    the view, and applies the augmentations. This class converts the
    arrays to tensors. It can also change the class order and the
    keypoint order of each task.

    Example:
        The ``loader`` section of a config that reads an existing
        dataset:

        .. code-block:: yaml

            loader:
              name: LuxonisLoaderTorch
              params:
                dataset_name: coco_test

        The ``loader`` section of a config that parses a directory into
        a new dataset:

        .. code-block:: yaml

            loader:
              name: LuxonisLoaderTorch
              params:
                dataset_dir: data/my_dataset
                dataset_name: my_dataset

    """

    def __init__(
        self,
        dataset_name: str | None = None,
        dataset_dir: str | None = None,
        dataset_type: DatasetType | None = None,
        team_id: str | None = None,
        bucket_type: Literal["internal", "external"] = "internal",
        bucket_storage: Literal["local", "s3", "gcs", "azure"] = "local",
        update_mode: Literal["all", "missing"] = "all",
        delete_existing: bool = True,
        filter_task_names: list[str] | None = None,
        min_bbox_visibility: float = 0.0,
        bbox_area_threshold: float = 0.0004,
        class_order_per_task: dict[str, list[str]] | None = None,
        kpts_mapping_per_task: dict[str, list[int]] | None = None,
        return_sample_metadata: bool = False,
        **kwargs,
    ):
        """Initialize the dataset and the ``LuxonisLoader``.

        With ``dataset_dir``, the loader parses the directory into a new
        dataset, or opens an existing local dataset, see
        ``delete_existing``. Without it, the loader opens the dataset
        ``dataset_name``. The ``LuxonisLoader`` downloads a remote
        dataset when it starts.

        Args:
            dataset_name (str | None): The name of the dataset. Without
                ``dataset_dir``, the loader opens this dataset. With
                ``dataset_dir``, the parsed dataset gets this name;
                ``None`` uses the directory name.
            dataset_dir (str | None): The directory to parse, in a format
                that ``LuxonisParser`` recognizes. It can be a local path,
                a remote URL, or a ZIP file. The parser downloads a
                remote directory to ``data/`` in the working directory.
            dataset_type (``DatasetType | None``): The format of
                ``dataset_dir``. ``None`` lets the parser detect it.
            team_id (str | None): The team ID of the dataset. It selects
                its local and remote location. ``None`` uses the
                ``LUXONISML_TEAM_ID`` setting of ``luxonis_ml``.
            bucket_type (``Literal["internal", "external"]``): The bucket
                type of a remote dataset. The loader uses it only without
                ``dataset_dir``.
            bucket_storage (``Literal["local", "s3", "gcs", "azure"]``):
                The storage backend of the dataset.
            update_mode (``Literal["all", "missing"]``): The sync mode for
                the media files of a remote dataset. ``"all"`` downloads
                all media files again. ``"missing"`` downloads only the
                media files that are not local. For a remote dataset, the
                ``LuxonisLoader`` always downloads the annotations and the
                metadata.
            delete_existing (bool): What to do when ``dataset_dir`` is set
                and a local dataset with the same name exists. ``True``
                logs a warning, deletes the existing dataset, and parses
                the directory again. ``False`` opens the existing dataset
                and does not parse.
            filter_task_names (list[str] | None): The names of the tasks
                to load. ``None`` loads all tasks. A name that is not in
                the dataset makes ``LuxonisLoader`` raise ``ValueError``.
            min_bbox_visibility (float): The minimum fraction of a box
                that must stay visible after the augmentations.
            bbox_area_threshold (float): The minimum area of a box,
                relative to the image area, in ``[0, 1]``. The
                augmentations remove a smaller box and the labels of its
                instance, such as the keypoints and the instance mask.
            class_order_per_task (dict[str, list[str]] | None): The class
                names of each task in their desired order. Each list
                must contain exactly the classes of its task. ``None``
                keeps the dataset order.
            kpts_mapping_per_task (dict[str, list[int]] | None): A new
                keypoint order for each task. For a list ``m``, the
                keypoint at position ``j`` is the original keypoint
                ``m[j]``. Each list must map every keypoint of its task.
                ``None`` keeps the original order.
            return_sample_metadata (bool): Whether ``__getitem__``
                returns the sample metadata as a third element.
            **kwargs (``Any``): Arguments for `BaseLoaderTorch`, such as
                ``view``, ``height``, and ``width``. This loader needs
                ``view``, and values other than ``None`` for ``height``,
                ``width``, and ``augmentation_config``.

        Raises:
            ValueError: If both ``dataset_dir`` and ``dataset_name`` are
                ``None``, or if ``height``, ``width``, or
                ``augmentation_config`` is ``None``.
            KeyError: If ``kpts_mapping_per_task`` has a task that is not
                in the dataset, or a task without keypoint labels.

        """
        super().__init__(**kwargs)
        self._return_sample_metadata = return_sample_metadata
        self.dataset = self._load_dataset(
            dataset_dir,
            dataset_name,
            dataset_type,
            delete_existing,
            team_id,
            bucket_type,
            bucket_storage,
        )
        if class_order_per_task is not None:
            self.dataset.set_class_order_per_task(class_order_per_task)

        self._validate_keypoint_mapping(kpts_mapping_per_task)
        self.kpts_mapping_per_task = kpts_mapping_per_task

        self.loader = LuxonisLoader(
            dataset=self.dataset,
            view=self.view,
            augmentation_engine=self.augmentation_engine,
            augmentation_config=[
                aug.model_dump(exclude={"active"})
                for aug in self.augmentation_config
            ],
            height=self.height,
            width=self.width,
            keep_aspect_ratio=self.keep_aspect_ratio,
            color_space=self.color_space,
            update_mode=update_mode,
            filter_task_names=filter_task_names,
            min_bbox_visibility=min_bbox_visibility,
            bbox_area_threshold=bbox_area_threshold,
            seed=self.seed,
        )

    def _load_dataset(
        self,
        dataset_dir: str | None,
        dataset_name: str | None,
        dataset_type: DatasetType | None,
        delete_existing: bool,
        team_id: str | None,
        bucket_type: Literal["internal", "external"],
        bucket_storage: Literal["local", "s3", "gcs", "azure"],
    ) -> LuxonisDataset:
        if dataset_dir is not None:
            return self._parse_dataset(
                dataset_dir, dataset_name, dataset_type, delete_existing
            )
        if dataset_name is None:
            raise ValueError(
                "Either `dataset_dir` or `dataset_name` must be provided."
            )
        return LuxonisDataset(
            dataset_name=dataset_name,
            team_id=team_id,
            bucket_type=bucket_type,
            bucket_storage=bucket_storage,
        )

    def _validate_keypoint_mapping(
        self, kpts_mapping_per_task: dict[str, list[int]] | None
    ) -> None:
        if kpts_mapping_per_task is None:
            return
        dataset_tasks = self.dataset.get_tasks()
        for task, new_mapping in kpts_mapping_per_task.items():
            self._validate_keypoint_task(task, dataset_tasks)
            if len(new_mapping) != len(set(new_mapping)):
                logger.warning(
                    f"Duplicate indices detected in keypoint mapping for task `{task}`. Verify that training on repeated keypoints is intentional."
                )

    @staticmethod
    def _validate_keypoint_task(
        task: str, dataset_tasks: Mapping[str, list[str]]
    ) -> None:
        if task not in dataset_tasks:
            raise KeyError(
                f"Task `{task}` specified in kpts_mapping_per_task but not present in dataset tasks ({list(dataset_tasks.keys())})"
            )
        if "keypoints" not in dataset_tasks[task]:
            raise KeyError(
                f"Task `{task}` specified in kpts_mapping_per_task but this task doesn't have `keypoints` annotations"
            )

    @override
    def __len__(self) -> int:
        return len(self.loader)

    @property
    @override
    def input_shapes(self) -> dict[str, Size]:
        """The shape ``[C, H, W]`` of the input image, keyed by
        ``image_source``.

        The dictionary has only the ``image_source`` key, also for a
        dataset with more than one image source. Each access loads the
        first sample, with the augmentations.

        """
        img = self[0][0]
        if isinstance(img, dict):
            img = img[self.image_source]
        return {self.image_source: img.shape}

    @override
    def __getitem__(
        self, idx: int
    ) -> (
        tuple[dict[str, Tensor] | Tensor, Labels]
        | tuple[dict[str, Tensor] | Tensor, Labels, Params]
    ):
        """Load a sample and convert it to tensors.

        The ``LuxonisLoader`` reads the sample and applies the
        augmentations. Then the method changes the keypoint order with
        ``kpts_mapping_per_task``, when it is set. It converts each array
        to a ``torch.float32`` tensor, and moves the image channels
        first. A string label becomes the character codes of its first
        string.

        Args:
            idx (int): The index of the sample.

        Returns:
            ``tuple[dict[str, Tensor] | Tensor, Labels] | tuple[dict[str, Tensor] | Tensor, Labels, Params]``:
            The image, the labels, and the sample metadata when
            ``return_sample_metadata`` is ``True``:

            - The image is a tensor of shape ``[C, H, W]`` when the
              dataset has one image source. For more sources, it is a
              dictionary that maps each source name to its image.
            - The labels map each ``"task_name/task_type"`` key to a
              tensor.
            - The sample metadata is the ``metadata`` of the
              ``LuxonisLoader`` output. It holds the metadata of the
              dataset record. The ``LuxonisLoader`` adds the
              ``"filenames"`` and ``"augmentations"`` keys, unless the
              record has them. A batch augmentation that merges more
              than one sample also adds the
              ``"batch_augmentation_metadata"`` key.

        Raises:
            ValueError: If a mapping in ``kpts_mapping_per_task`` does
                not have one index for each keypoint of its task, and
                the sample has keypoints of that task.

        """
        output = self.loader[idx]
        img, labels = output
        if isinstance(img, np.ndarray):
            img = {self.image_source: img}

        if self.kpts_mapping_per_task is not None:
            labels = self._remap_keypoints(labels)

        img = {k: self.img_numpy_to_torch(v) for k, v in img.items()}
        if len(img) == 1:
            img = next(iter(img.values()))

        tensor_labels = self.dict_numpy_to_torch(labels)
        if self._return_sample_metadata:
            return img, tensor_labels, output.metadata
        return img, tensor_labels

    def _remap_keypoints(
        self, labels: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Change the keypoint order of each task in
        ``kpts_mapping_per_task``.

        The method skips a task without keypoint labels in the sample,
        and a task with no instances. It changes ``labels`` in place.

        Args:
            labels (``dict[str, np.ndarray]``): The labels of one sample.
                A ``"task_name/keypoints"`` array has the shape
                ``[N, 3 * K]``.

        Returns:
            ``dict[str, np.ndarray]``: The same ``labels`` dictionary.

        Raises:
            ValueError: If a mapping does not have ``K`` indices.

        """
        for task, new_mapping in self.kpts_mapping_per_task.items():  # type: ignore
            key = f"{task}/keypoints"
            if key not in labels:
                continue

            original = labels[key]
            if original.size == 0:
                continue

            n_samples, flat_dim = original.shape
            kpts = original.reshape(n_samples, -1, 3)

            expected, got = kpts.shape[1], len(new_mapping)
            if expected != got:
                raise ValueError(
                    f"Invalid keypoint mapping for task '{task}': expected {expected} indices, got {got}."
                )

            labels[key] = kpts[:, new_mapping, :].reshape(n_samples, flat_dim)

        return labels

    @override
    def get_classes(self) -> dict[str, dict[str, int]]:
        """Return the class names and the class IDs of each task.

        The mapping comes from the ``LuxonisLoader``. With
        ``filter_task_names``, it has only those tasks. It has the class
        order from ``class_order_per_task``.

        The ``LuxonisLoader`` adds a ``"background"`` class with the ID
        ``0`` to a segmentation task when all of these are true:

        - A sample of the view has masks of the task, and the masks do
          not cover the whole image.
        - The task has more than one class.
        - The task has no ``"background"`` class.

        The IDs of the other classes then increase by ``1``. Thus the
        loaders of two views can return different mappings.

        Returns:
            dict[str, dict[str, int]]: The class name to class ID
            mapping of each task, keyed by the task name.

        """
        return self.loader._classes

    @override
    def get_n_keypoints(self) -> dict[str, int]:
        """Return the number of keypoints of each task.

        The count is the number of keypoint names in the skeleton of the
        task, from the dataset metadata. A task without a skeleton is not
        in the result.

        Returns:
            dict[str, int]: The number of keypoints, keyed by the task
            name.

        """
        skeletons = self.dataset.get_skeletons()
        return {task: len(skeletons[task][0]) for task in skeletons}

    @override
    def get_metadata_types(
        self,
    ) -> dict[str, type[int] | type[Category] | type[float] | type[str]]:
        """Return the Python type of each metadata label.

        The dataset stores the name of the type. The method maps
        ``"float"``, ``"int"``, and ``"str"`` to the type of that name. It
        maps ``"Category"`` to ``int``, because the labels of the loader
        hold the integer code of a category.

        Returns:
            ``dict[str, type[int] | type[Category] | type[float] | type[str]]``:
            The type of each metadata label, keyed by the label name,
            such as ``"task_name/metadata/color"``.

        """
        return {
            k: {"float": float, "int": int, "str": str, "Category": int}[v]
            for k, v in self.dataset.get_metadata_types().items()
        }

    @override
    def get_categorical_encodings(self) -> dict[str, dict[str, int]]:
        """Return the integer code of each category of each metadata
        label.

        The codes come from the dataset metadata. The labels of the
        loader hold these codes in place of the category names.

        Returns:
            dict[str, dict[str, int]]: The category to code mapping of
            each categorical metadata label, keyed by the label name,
            such as ``"task_name/metadata/color"``.

        """
        return self.dataset.get_categorical_encodings()

    @override
    def augment_test_image(self, img: dict[str, Tensor] | Tensor) -> Tensor:
        """Apply the augmentations of the loader to one raw image.

        Inference calls this method to prepare an image like the samples
        of the view. The augmentations get one sample with no labels.
        They include the resize to ``height`` and ``width``.

        Args:
            img (``dict[str, Tensor] | Tensor``): The image of shape
                ``[H, W, C]``. A dictionary maps each source name to its
                image. A tensor is the image of ``image_source``.

        Returns:
            ``Tensor``: The augmented image of shape
            ``[height, width, C]``. For a dictionary with more than one
            source, it is the first image of the augmented dictionary.
            When the ``LuxonisLoader`` has no augmentations, the method
            returns the ``image_source`` image with no change. The
            ``LuxonisLoader`` that ``__init__`` builds always has
            augmentations.

        """
        if isinstance(img, Tensor):
            img = {self.image_source: img}

        if self.loader._augmentations is None:
            return img[self.image_source]
        img_arr = {k: v.numpy() for k, v in img.items()}
        augmented_dict = self.loader._augmentations.apply([(img_arr, {})])[0]
        return torch.tensor(next(iter(augmented_dict.values())))

    def _parse_dataset(
        self,
        dataset_dir: str,
        dataset_name: str | None,
        dataset_type: DatasetType | None,
        delete_existing: bool,
    ) -> LuxonisDataset:
        if dataset_name is None:
            dataset_name = Path(dataset_dir).name
        if LuxonisDataset.exists(dataset_name):
            if not delete_existing:
                return LuxonisDataset(dataset_name=dataset_name)
            logger.warning(
                f"Dataset '{dataset_name}' already exists. "
                "The dataset will be generated again to ensure "
                "the latest data are used. If you don't want to regenerate "
                "the dataset each time, set `delete_existing` to `False`"
            )

        if dataset_type is None:
            logger.warning(
                "Dataset type is not set. "
                "Attempting to infer it from the directory structure. "
                "If this fails, please set the dataset type manually. "
                f"Supported types are: {list(DatasetType.__members__)}."
            )

        logger.info(
            f"Parsing dataset from {dataset_dir} with name '{dataset_name}'"
        )

        return LuxonisParser(
            dataset_dir,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            save_dir="data",
            delete_local=True,
            delete_remote=True,
        ).parse()
