import random
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.utils import LuxonisFileSystem
from torch import Tensor

from .base_loader import LuxonisLoaderTorchOutput
from .luxonis_loader_torch import LuxonisLoaderTorch
from .perlin import apply_anomaly_to_img


class LuxonisLoaderPerlinNoise(LuxonisLoaderTorch):
    def __init__(
        self,
        *args,
        anomaly_source_path: str,
        noise_prob: float = 0.5,
        **kwargs,
    ):
        """Custom loader for Luxonis datasets that adds Perlin noise
        during training with a given probability.

        @param anomaly_source_path: Path to the anomaly dataset from
            where random samples are drawn for noise.
        @param noise_prob: The probability with which to apply Perlin
            noise (only used during training).
        """
        super().__init__(*args, **kwargs)

        try:
            self.anomaly_source_path = LuxonisFileSystem.download(
                anomaly_source_path, dest="./data"
            )
        except Exception as e:
            raise FileNotFoundError(
                "The anomaly source path is invalid."
            ) from e

        from luxonis_train.core.utils.infer_utils import IMAGE_FORMATS

        self.anomaly_files = [
            f
            for f in self.anomaly_source_path.rglob("*")
            if f.suffix.lower() in IMAGE_FORMATS
        ]
        if not self.anomaly_files:
            raise FileNotFoundError(
                "No image files found at the specified path."
            )

        self.noise_prob = noise_prob
        if len(self.loader.dataset.get_tasks()) > 1:
            # TODO: Can be extended to multiple tasks
            raise ValueError(
                "This loader only supports datasets with a single task."
            )
        self.task_name = next(iter(self.loader.dataset.get_tasks()))

        augmentations = cast(AlbumentationsEngine, self.loader.augmentations)
        if augmentations is None or augmentations.pixel_transform is None:
            self.pixel_augs = None
        else:
            self.pixel_augs = augmentations.pixel_transform

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        img, labels = self.loader[idx]

        img = np.transpose(img, (2, 0, 1))
        tensor_img = Tensor(img)

        if self.view[0] == "train" and random.random() < self.noise_prob:
            aug_tensor_img, an_mask = apply_anomaly_to_img(
                tensor_img,
                anomaly_source_paths=self.anomaly_files,
                pixel_augs=self.pixel_augs,
            )
        else:
            aug_tensor_img = tensor_img
            h, w = aug_tensor_img.shape[-2:]
            an_mask = torch.zeros((h, w))

        tensor_labels = {f"{self.task_name}/original/segmentation": tensor_img}
        for task, array in labels.items():
            tensor_labels[task] = Tensor(array)

        tensor_labels[f"{self.task_name}/segmentation"] = (
            F.one_hot(an_mask.long(), 2).permute(2, 0, 1).float()
        )

        return {self.image_source: aug_tensor_img}, tensor_labels
