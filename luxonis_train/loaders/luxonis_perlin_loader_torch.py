import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from luxonis_ml.utils import LuxonisFileSystem
from torch import Tensor

from luxonis_train.enums import TaskType

from .base_loader import LuxonisLoaderTorchOutput
from .luxonis_loader_torch import LuxonisLoaderTorch
from .perlin import apply_anomaly_to_img


class LuxonisLoaderPerlinNoise(LuxonisLoaderTorch):
    def __init__(
        self,
        *args,
        anomaly_source_path: str | None = None,
        noise_prob: float = 0.5,
        **kwargs,
    ):
        """Custom loader for Luxonis datasets that adds Perlin noise
        during training with a given probability.

        @type anomaly_source_path: str | None
        @param anomaly_source_path: Path to the anomaly dataset from
            where random samples are drawn for noise.
        @type noise_prob: int
        @param noise_prob: The probability with which to apply Perlin
            noise (only used during training). normalization.
        """
        super().__init__(*args, **kwargs)
        lux_fs = LuxonisFileSystem(path=anomaly_source_path)
        if lux_fs.protocol in ["s3", "gcs"]:
            anomaly_source_path = lux_fs.get_dir(
                remote_path=anomaly_source_path, local_path="./data"
            )
        else:
            anomaly_source_path = lux_fs.path

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
        self.base_loader.add_background = True
        self.base_loader.class_mappings["segmentation"]["background"] = 0
        self.base_loader.class_mappings["segmentation"] = {
            k: (v + 1 if k != "background" else v)
            for k, v in self.base_loader.class_mappings["segmentation"].items()
        }

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        img, labels = self.base_loader[idx]

        img = np.transpose(img, (2, 0, 1))
        tensor_img = Tensor(img)

        if (
            self.view[0] == "train"
            and self.anomaly_source_path is not None
            and random.random() < self.noise_prob
        ):
            aug_tensor_img, an_mask = apply_anomaly_to_img(
                tensor_img,
                anomaly_source_paths=self.anomaly_source_paths,
                pixel_augs=self.augmentations.pixel_transform.transforms,
            )
        else:
            aug_tensor_img = tensor_img
            h, w = aug_tensor_img.shape[-2:]
            an_mask = torch.zeros((h, w))

        tensor_labels = {"original": (tensor_img, TaskType.ARRAY)}
        if self.view[0] == "train":
            tensor_labels["segmentation"] = (
                F.one_hot(an_mask.long(), 2).permute(2, 0, 1).float(),
                TaskType.SEGMENTATION,
            )
        else:
            for task, (array, label_type) in labels.items():
                tensor_labels[task] = (
                    Tensor(array),
                    TaskType(label_type.value),
                )

        return {self.image_source: aug_tensor_img}, tensor_labels
