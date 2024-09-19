import json
import logging
import random
import warnings
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from luxonis_ml.data import (
    Augmentations,
    BucketStorage,
    BucketType,
    LuxonisDataset,
    LuxonisLoader,
)
from luxonis_ml.data.loaders.base_loader import LuxonisLoaderOutput
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.data.utils.enums import LabelType
from luxonis_ml.enums import DatasetType
from torch import Size, Tensor
from typeguard import typechecked

from .base_loader import BaseLoaderTorch, LuxonisLoaderTorchOutput

logger = logging.getLogger(__name__)


class OBBLoaderTorch(BaseLoaderTorch):
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
        """Torch-compatible loader for Luxonis datasets for obb.

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

        self.instances = []
        splits_path = self.dataset.metadata_path / "splits.json"
        if not splits_path.exists():
            raise RuntimeError(
                "Cannot find splits! Ensure you call dataset.make_splits()"
            )
        with open(splits_path, "r") as file:
            splits = json.load(file)

        for view in self.view:
            self.instances.extend(splits[view])

        self.base_loader = OBBLoader(
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
        tensor_labels = {}
        for task, (array, label_type) in labels.items():
            tensor_labels[task] = (Tensor(array), label_type)

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
            dataset_name = Path(dataset_dir).stem
            if dataset_type is not None:
                dataset_name += f"_{dataset_type.value}"

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


class OBBLoader(LuxonisLoader):
    def __init__(
        self,
        dataset: LuxonisDataset,
        view: Union[str, List[str]] = "train",
        stream: bool = False,
        augmentations: Optional[Augmentations] = None,
        *,
        force_resync: bool = False,
    ) -> None:
        """A loader class used for loading data from L{LuxonisDataset}
        for oriented bounding boxes.

        @type dataset: LuxonisDataset
        @param dataset: LuxonisDataset to use
        @type view: Union[str, List[str]]
        @param view: What splits to use. Can be either a single split or
            a list of splits. Defaults to "train".
        @type stream: bool
        @param stream: Flag for data streaming. Defaults to C{False}.
        @type augmentations: Optional[luxonis_ml.loader.Augmentations]
        @param augmentations: Augmentation class that performs
            augmentations. Defaults to C{None}.
        @type force_resync: bool
        @param force_resync: Flag to force resync from cloud. Defaults
            to C{False}.
        """
        super().__init__(
            dataset=dataset,
            view=view,
            stream=stream,
            augmentations=augmentations,
            force_resync=force_resync,
        )

    def __getitem__(self, idx: int) -> LuxonisLoaderOutput:
        """Function to load a sample consisting of an image and its
        annotations.

        @type idx: int
        @param idx: The (often random) integer index to retrieve a
            sample from the dataset.
        @rtype: LuxonisLoaderOutput
        @return: The loader ouput consisting of the image and a
            dictionary defining its annotations.
        """

        if self.augmentations is None:
            return self._load_image_with_annotations(idx)

        indices = [idx]
        if self.augmentations.is_batched:
            other_indices = [i for i in range(len(self)) if i != idx]
            if self.augmentations.aug_batch_size > len(self):
                warnings.warn(
                    f"Augmentations batch_size ({self.augmentations.aug_batch_size}) is larger than dataset size ({len(self)}), samples will include repetitions."
                )
                random_fun = random.choices
            else:
                random_fun = random.sample
            picked_indices = random_fun(
                other_indices, k=self.augmentations.aug_batch_size - 1
            )
            indices.extend(picked_indices)

        out_dict: Dict[str, Tuple[np.ndarray, LabelType]] = {}
        loaded_anns = [self._load_image_with_annotations(i) for i in indices]
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        while loaded_anns[0][1]:
            aug_input_data = []
            label_to_task = {}
            nk = 0
            ns = 0
            for img, annotations in loaded_anns:
                label_dict: Dict[LabelType, np.ndarray] = {}
                task_dict: Dict[LabelType, str] = {}
                for task in sorted(list(annotations.keys())):
                    array, label_type = annotations[task]
                    if label_type not in label_dict:
                        # ensure that bounding box annotations are added to the
                        # `label_dict` before keypoints
                        if label_type == LabelType.KEYPOINTS:
                            if (
                                LabelType.BOUNDINGBOX
                                in map(
                                    itemgetter(1), list(annotations.values())
                                )
                                and LabelType.BOUNDINGBOX not in label_dict  # type: ignore
                            ):
                                continue

                            if (
                                LabelType.BOUNDINGBOX in label_dict  # type: ignore
                                and LabelType.BOUNDINGBOX
                                in map(
                                    itemgetter(1), list(annotations.values())
                                )
                            ):
                                bbox_task = task_dict[LabelType.BOUNDINGBOX]
                                *_, bbox_suffix = bbox_task.split("-", 1)
                                *_, kp_suffix = task.split("-", 1)
                                if bbox_suffix != kp_suffix:
                                    continue

                        label_dict[label_type] = array
                        label_to_task[label_type] = task
                        task_dict[label_type] = task
                        annotations.pop(task)
                        if label_type == LabelType.KEYPOINTS:
                            nk = (array.shape[1] - 1) // 3
                        if label_type == LabelType.SEGMENTATION:
                            ns = array.shape[0]

                aug_input_data.append((img, label_dict))

            # NOTE: To ensure the same augmentation is applied to all samples
            # in case of multiple tasks per LabelType
            random.setstate(random_state)
            np.random.set_state(np_random_state)

            # NOTE: consider implementing resizing using the aspect ratio of the original input images
            # height, width = img.shape[0], img.shape[1]
            # # Determine the larger dimension
            # if height > width:
            #     aspect_ratio = round(height / width, 2)
            #     new_height = 640
            #     new_width = round(int(640 / aspect_ratio), -1)
            # else:
            #     aspect_ratio = round(width / height, 2)
            #     new_width = 640
            #     new_height = round(int(640 / aspect_ratio), -1)

            # img_resized = cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_AREA)

            # NOTE: Temporary solution, to demonstrate training functionality oh the DOTA dataset.
            # If it's needed can be changed to the size from config file
            img_resized = cv2.resize(
                img, (512, 512), interpolation=cv2.INTER_AREA
            )
            img_norm = img_resized / 255  # [0, 1]

            img, aug_annotations = self.augmentations(
                aug_input_data, nk=nk, ns=ns
            )
            for label_type, array in aug_annotations.items():
                out_dict[label_to_task[label_type]] = (array, label_type)

        return img_norm, out_dict  # type: ignore
