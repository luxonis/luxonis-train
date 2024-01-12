import numpy as np
from luxonis_ml.data import Augmentations, LuxonisDataset, LuxonisLoader
from torch import Size, Tensor

from .base_loader import BaseLoaderTorch, LuxonisLoaderTorchOutput


class LuxonisLoaderTorch(BaseLoaderTorch):
    def __init__(
        self,
        dataset: LuxonisDataset,
        view: str = "train",
        stream: bool = False,
        augmentations: Augmentations | None = None,
    ):
        self.base_loader = LuxonisLoader(
            dataset=dataset,
            view=view,
            stream=stream,
            augmentations=augmentations,
        )

    def __len__(self) -> int:
        return len(self.base_loader)

    @property
    def input_shape(self) -> Size:
        img, _ = self[0]
        return Size([1, *img.shape])

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        img, annotations = self.base_loader[idx]

        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        tensor_img = Tensor(img)
        for key in annotations:
            annotations[key] = Tensor(annotations[key])  # type: ignore

        return tensor_img, annotations
