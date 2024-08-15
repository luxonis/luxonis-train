from typing import Literal

import numpy as np
from luxonis_ml.data import BucketStorage, BucketType, LuxonisDataset, LuxonisLoader
from torch import Size, Tensor

from .base_loader import BaseLoaderTorch, LuxonisLoaderTorchOutput


class LuxonisLoaderTorch(BaseLoaderTorch):
    def __init__(
        self,
        dataset_name: str,
        team_id: str | None = None,
        bucket_type: Literal["internal", "external"] = "internal",
        bucket_storage: Literal["local", "s3", "gcs", "azure"] = "local",
        stream: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
