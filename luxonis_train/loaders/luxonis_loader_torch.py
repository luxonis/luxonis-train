import logging
from typing import Literal

import numpy as np
from luxonis_ml.data import (
    Augmentations,
    BucketStorage,
    BucketType,
    LuxonisDataset,
    LuxonisLoader,
)
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.enums import DatasetType
from torch import Size, Tensor
from typeguard import typechecked

from luxonis_train.enums import TaskType

from .base_loader import BaseLoaderTorch, LuxonisLoaderTorchOutput

logger = logging.getLogger(__name__)


class LuxonisLoaderTorch(BaseLoaderTorch):
    @typechecked
    def __init__(
        self,
        dataset_name: str | None = None,
        dataset_dir: str | None = None,
        dataset_type: DatasetType | None = None,
        team_id: str | None = None,
        bucket_type: Literal["internal", "external"] = "internal",
        bucket_storage: Literal["local", "s3", "gcs", "azure"] = "local",
        stream: bool = False,
        delete_existing: bool = True,
        view: str | list[str] = "train",
        augmentations: Augmentations | None = None,
        **kwargs,
    ):
        """Torch-compatible loader for Luxonis datasets.

        Can either use an already existing dataset or parse a new one from a directory.

        @type dataset_name: str | None
        @param dataset_name: Name of the dataset to load. If not provided, the
            C{dataset_dir} argument must be provided instead. If both C{dataset_dir} and
            C{dataset_name} are provided, the dataset will be parsed from the directory
            and saved with the provided name.
        @type dataset_dir: str | None
        @param dataset_dir: Path to the dataset directory. It can be either a local path
            or a URL. The data can be in a zip file. If not provided, C{dataset_name} of
            an existing dataset must be provided.
        @type dataset_type: str | None
        @param dataset_type: Type of the dataset. Only relevant when C{dataset_dir} is
            provided. If not provided, the type will be inferred from the directory
            structure.
        @type team_id: str | None
        @param team_id: Optional unique team identifier for the cloud.
        @type bucket_type: Literal["internal", "external"]
        @param bucket_type: Type of the bucket. Only relevant for remote datasets.
            Defaults to 'internal'.
        @type bucket_storage: Literal["local", "s3", "gcs", "azure"]
        @param bucket_storage: Type of the bucket storage. Defaults to 'local'.
        @type stream: bool
        @param stream: Flag for data streaming. Defaults to C{False}.
        @type delete_existing: bool
        @param delete_existing: Only relevant when C{dataset_dir} is provided. By
            default, the dataset is parsed again every time the loader is created
            because the underlying data might have changed. If C{delete_existing} is set
            to C{False} and a dataset of the same name already exists, the existing
            dataset will be used instead of re-parsing the data.
        @type view: str | list[str]
        @param view: A single split or a list of splits that will be used to create a
            view of the dataset. Each split is a string that represents a subset of the
            dataset. The available splits depend on the dataset, but usually include
            'train', 'val', and 'test'. Defaults to 'train'.
        @type augmentations: Augmentations | None
        @param augmentations: Augmentations to apply to the data. Defaults to C{None}.
        """
        super().__init__(view=view, augmentations=augmentations, **kwargs)
        if dataset_dir is not None:
            self.dataset = self._parse_dataset(
                dataset_dir, dataset_name, dataset_type, delete_existing
            )
        else:
            if dataset_name is None:
                raise ValueError(
                    "Either `dataset_dir` or `dataset_name` must be provided."
                )
            self.dataset = LuxonisDataset(
                dataset_name=dataset_name,
                team_id=team_id,
                bucket_type=BucketType(bucket_type),
                bucket_storage=BucketStorage(bucket_storage),
            )
        self.base_loader = LuxonisLoader(
            dataset=self.dataset,
            view=self.view,
            stream=stream,
            augmentations=self.augmentations,
        )

    def __len__(self) -> int:
        return len(self.base_loader)

    @property
    def input_shapes(self) -> dict[str, Size]:
        img = self[0][0][self.image_source]
        return {self.image_source: img.shape}

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        img, labels = self.base_loader[idx]

        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        tensor_img = Tensor(img)
        tensor_labels: dict[str, tuple[Tensor, TaskType]] = {}
        for task, (array, label_type) in labels.items():
            tensor_labels[task] = (Tensor(array), TaskType(label_type.value))

        return {self.image_source: tensor_img}, tensor_labels

    def get_classes(self) -> dict[str, list[str]]:
        _, classes = self.dataset.get_classes()
        return {task: classes[task] for task in classes}

    def get_n_keypoints(self) -> dict[str, int]:
        skeletons = self.dataset.get_skeletons()
        return {task: len(skeletons[task][0]) for task in skeletons}

    def _parse_dataset(
        self,
        dataset_dir: str,
        dataset_name: str | None,
        dataset_type: DatasetType | None,
        delete_existing: bool,
    ) -> LuxonisDataset:
        if dataset_name is None:
            dataset_name = dataset_dir.split("/")[-1]

        if LuxonisDataset.exists(dataset_name):
            if not delete_existing:
                return LuxonisDataset(dataset_name=dataset_name)
            else:
                logger.warning(
                    f"Dataset {dataset_name} already exists. "
                    "The dataset will be generated again to ensure the latest data are used. "
                    "If you don't want to regenerate the dataset every time, set `delete_existing=False`'"
                )

        if dataset_type is None:
            logger.warning(
                "Dataset type is not set. "
                "Attempting to infer it from the directory structure. "
                "If this fails, please set the dataset type manually. "
                f"Supported types are: {', '.join(DatasetType.__members__)}."
            )

        logger.info(
            f"Parsing dataset from {dataset_dir} with name '{dataset_name}'"
        )

        return LuxonisParser(
            dataset_dir,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            save_dir="data",
            delete_existing=True,
        ).parse()
