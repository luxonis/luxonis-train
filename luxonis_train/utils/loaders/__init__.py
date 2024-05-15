from luxonis_ml.data import LuxonisDataset

from luxonis_train.utils.registry import DATASETS

from .base_loader import (
    BaseLoaderTorch,
    LuxonisLoaderTorchOutput,
    collate_fn,
)
from .luxonis_loader_torch import LuxonisLoaderTorch

DATASETS.register_module(module=LuxonisDataset)


__all__ = [
    "LuxonisLoaderTorch",
    "collate_fn",
    "BaseLoaderTorch",
    "LuxonisLoaderTorchOutput",
]
