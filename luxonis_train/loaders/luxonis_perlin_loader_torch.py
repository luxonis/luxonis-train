import glob
import os
import random

import numpy as np
import torch
from torch import Tensor

from luxonis_train.enums import TaskType

from .luxonis_loader_torch import LuxonisLoaderTorch
from .perlin import apply_anomaly_to_img


class LuxonisLoaderPerlinNoise(LuxonisLoaderTorch):
    def __init__(
        self,
        *args,
        anomaly_source_path: str | None = None,
        noise_prob: int = 0.5,
        mean: list | None = None,
        std: list | None = None,
        **kwargs,
    ):
        """Custom loader for Luxonis datasets that adds Perlin noise
        during training with a given probability.

        @type anomaly_source_path: str | None
        @param anomaly_source_path: Path to the anomaly dataset from
            where random samples are drawn for noise.
        @type noise_prob: int
        @param noise_prob: The probability with which to apply Perlin
            noise (only used during training).
        @type mean: list
        @param mean: The mean values for the image normalization.
        @type std: list
        @param std: The standard deviation values for the image
            normalization.
        """
        super().__init__(*args, **kwargs)
        if anomaly_source_path and os.path.exists(anomaly_source_path):
            self.anomaly_source_paths = sorted(
                glob.glob(os.path.join(anomaly_source_path, "*/*.jpg"))
            )
            if not self.anomaly_source_paths:
                raise FileNotFoundError(
                    "No .jpg files found at the specified path."
                )
        else:
            raise ValueError("Invalid or unspecified anomaly source path.")
        self.anomaly_source_path = anomaly_source_path
        self.noise_prob = noise_prob
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

    def __getitem__(self, idx: int) -> dict:
        img, labels = self.base_loader[idx]

        img = np.transpose(img, (2, 0, 1))
        tensor_img = Tensor(img)

        if (
            self.view[0] == "train"
            and self.anomaly_source_path is not None
            and random.random() < self.noise_prob
        ):
            tensor_img, an_mask = apply_anomaly_to_img(
                tensor_img,
                anomaly_source_paths=self.anomaly_source_paths,
                mean=self.mean,
                std=self.std,
            )
        else:
            h, w = tensor_img.shape[-2:]
            an_mask = torch.zeros((h, w))

        tensor_labels: dict[str, tuple[Tensor, TaskType]] = {}
        if self.view[0] == "train":
            tensor_labels["segmentation"] = (
                an_mask.unsqueeze(0),
                TaskType.SEGMENTATION,
            )
        else:
            for task, (array, label_type) in labels.items():
                tensor_labels[task] = (
                    Tensor(array),
                    TaskType(label_type.value),
                )

        return {self.image_source: tensor_img}, tensor_labels
