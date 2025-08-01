from pathlib import Path
from typing import Literal

import numpy as np
import torch
from loguru import logger
from luxonis_ml.data import Category, LuxonisDataset, LuxonisLoader
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.enums import DatasetType
from torch import Size, Tensor
from typing_extensions import override

from luxonis_train.typing import Labels

from .base_loader import BaseLoaderTorch


class LuxonisLoaderTorch(BaseLoaderTorch):
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
        seed: int | None = None,
        **kwargs,
    ):
        """Torch-compatible loader for Luxonis datasets.

        Can either use an already existing dataset or parse a new one
        from a directory.

        @type dataset_name: str | None
        @param dataset_name: Name of the dataset to load. If not
            provided, the C{dataset_dir} argument must be provided
            instead. If both C{dataset_dir} and C{dataset_name} are
            provided, the dataset will be parsed from the directory and
            saved with the provided name.
        @type dataset_dir: str | None
        @param dataset_dir: Path to the dataset directory. It can be
            either a local path or a URL. The data can be in a zip file.
            If not provided, C{dataset_name} of an existing dataset must
            be provided.
        @type dataset_type: str | None
        @param dataset_type: Type of the dataset. Only relevant when
            C{dataset_dir} is provided. If not provided, the type will
            be inferred from the directory structure.
        @type team_id: str | None
        @param team_id: Optional unique team identifier for the cloud.
        @type bucket_type: Literal["internal", "external"]
        @param bucket_type: Type of the bucket. Only relevant for remote
            datasets. Defaults to 'internal'.
        @type bucket_storage: Literal["local", "s3", "gcs", "azure"]
        @param bucket_storage: Type of the bucket storage. Defaults to
            'local'.
        @type update_mode: Literal["all", "missing"]
        @param update_mode: Enum that determines the sync mode for media files of the remote dataset (annotations and metadata are always overwritten):
            - UpdateMode.MISSING: Downloads only the missing media files for the dataset.
            - UpdateMode.ALL: Always downloads and overwrites all media files in the local dataset.
        @type delete_existing: bool
        @param delete_existing: Only relevant when C{dataset_dir} is
            provided. By default, the dataset is parsed again every time
            the loader is created because the underlying data might have
            changed. If C{delete_existing} is set to C{False} and a
            dataset of the same name already exists, the existing
            dataset will be used instead of re-parsing the data.
        @type filter_task_names: list[str] | None
        @param filter_task_names: List of task names to filter the
            dataset by. If provided, only the tasks with the specified
            names will be loaded. If not provided, all tasks will be
            loaded.
        @type min_bbox_visibility: float
        @param min_bbox_visibility: Minimum fraction of the original bounding box that must remain visible after augmentation.
        @type bbox_area_threshold: float
        @param bbox_area_threshold: Minimum area threshold for bounding boxes to be considered valid. In the range [0, 1].
            Default is 0.0004, which corresponds to a small area threshold to remove invalid bboxes and respective keypoints.
        @type class_order_per_task: dict[str, list[str]] | None
        @param class_order_per_task: Dictionary mapping task names to a list of class names.
            If provided, the classes for the specified tasks will be reordered.
        @type seed: Optional[int]
        @param seed: The random seed to use for the augmentations.
        """
        super().__init__(**kwargs)
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
                bucket_type=bucket_type,
                bucket_storage=bucket_storage,
            )
        if class_order_per_task is not None:
            self.dataset.set_class_order_per_task(class_order_per_task)
        self.loader = LuxonisLoader(
            dataset=self.dataset,
            view=self.view,
            augmentation_engine=self.augmentation_engine,
            augmentation_config=[
                aug.model_dump() for aug in self.augmentation_config
            ],
            height=self.height,
            width=self.width,
            keep_aspect_ratio=self.keep_aspect_ratio,
            color_space=self.color_space,
            update_mode=update_mode,
            filter_task_names=filter_task_names,
            min_bbox_visibility=min_bbox_visibility,
            bbox_area_threshold=bbox_area_threshold,
            seed=seed,
        )

    @override
    def __len__(self) -> int:
        return len(self.loader)

    @property
    @override
    def input_shapes(self) -> dict[str, Size]:
        img = self[0][0][self.image_source]
        return {self.image_source: img.shape}

    @override
    def get(self, idx: int) -> tuple[Tensor, Labels]:
        img, labels = self.loader[idx]

        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        tensor_img = torch.tensor(img)

        return tensor_img, self.dict_numpy_to_torch(labels)

    @override
    def get_classes(self) -> dict[str, dict[str, int]]:
        return self.dataset.get_classes()

    @override
    def get_n_keypoints(self) -> dict[str, int]:
        skeletons = self.dataset.get_skeletons()
        return {task: len(skeletons[task][0]) for task in skeletons}

    @override
    def get_metadata_types(
        self,
    ) -> dict[str, type[int] | type[Category] | type[float] | type[str]]:
        return {
            k: {"float": float, "int": int, "str": str, "Category": int}[v]
            for k, v in self.dataset.get_metadata_types().items()
        }

    @override
    def augment_test_image(self, img: dict[str, Tensor]) -> Tensor:
        if self.loader.augmentations is None:
            return img[self.image_source]
        img_arr = {k: v.numpy() for k, v in img.items()}
        augmented_dict = self.loader.augmentations.apply([(img_arr, {})])[0]
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
