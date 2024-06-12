from typing import Literal

import numpy as np
from luxonis_ml.data import (
    BucketStorage,
    BucketType,
    LuxonisDataset,
    LuxonisLoader,
)
from torch import Size, Tensor

from .base_loader import BaseLoaderTorch, LuxonisLoaderTorchOutput


class LuxonisLoaderTorch(BaseLoaderTorch):
    def __init__(
        self,
        dataset_name: str | None = None,
        team_id: str | None = None,
        dataset_id: str | None = None,
        bucket_type: Literal["internal", "external"] = "internal",
        bucket_storage: Literal["local", "s3", "gcs", "azure"] = "local",
        stream: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = LuxonisDataset(
            dataset_name=dataset_name,
            team_id=team_id,
            dataset_id=dataset_id,
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
    def input_shape(self) -> dict[str, Size]:
        img = self[0][0][self.images_name]
        return {self.images_name: img.shape}

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        img, labels = self.base_loader[idx]

        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        tensor_img = Tensor(img)
        tensor_labels = {}
        for task, (array, label_type) in labels.items():
            tensor_labels[task] = (Tensor(array), label_type)

        return {self.images_name: tensor_img}, tensor_labels

    def get_classes(self) -> dict[str, list[str]]:
        _, classes = self.dataset.get_classes()
        return {task: classes[task] for task in classes}

    def get_skeletons(self) -> dict[str, dict] | None:
        return self.dataset.get_skeletons()
