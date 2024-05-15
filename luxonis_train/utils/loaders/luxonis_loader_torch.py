import numpy as np
from luxonis_ml.data import LuxonisLoader
from torch import Size, Tensor

from .base_loader import BaseLoaderTorch, LuxonisLoaderTorchOutput


class LuxonisLoaderTorch(BaseLoaderTorch):
    def __init__(self, stream: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.base_loader = LuxonisLoader(
            dataset=self.dataset,
            view=self.view,
            stream=stream,
            augmentations=self.augmentations,
        )

    def __len__(self) -> int:
        return len(self.base_loader)

    @property
    def input_shape(self) -> Size:
        img, _ = self[0]
        return Size([1, *img.shape])

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        img, group_annotations = self.base_loader[idx]

        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        tensor_img = Tensor(img)
        for task in group_annotations:
            annotations = group_annotations[task]
            for key in annotations:
                annotations[key] = Tensor(annotations[key])  # type: ignore

        return tensor_img, group_annotations
