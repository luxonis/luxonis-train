from abc import ABC, abstractmethod

import torch
from luxonis_ml.data import Augmentations
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor
from torch.utils.data import Dataset

from luxonis_train.utils.registry import LOADERS
from luxonis_train.utils.types import Labels, LabelType

LuxonisLoaderTorchOutput = tuple[Tensor, Labels]
"""LuxonisLoaderTorchOutput is a tuple of images and corresponding labels."""


class BaseLoaderTorch(
    Dataset[LuxonisLoaderTorchOutput],
    ABC,
    metaclass=AutoRegisterMeta,
    register=False,
    registry=LOADERS,
):
    """Base abstract loader class that enforces LuxonisLoaderTorchOutput output label
    structure."""

    def __init__(
        self,
        view: str,
        augmentations: Augmentations | None = None,
    ):
        self.view = view
        self.augmentations = augmentations

    @property
    @abstractmethod
    def input_shape(self) -> Size:
        """Input shape in [N,C,H,W] format."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Returns length of the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        """Loads sample from dataset.

        @type idx: int
        @param idx: Sample index.
        @rtype: L{LuxonisLoaderTorchOutput}
        @return: Sample's data in L{LuxonisLoaderTorchOutput} format
        """
        ...

    @abstractmethod
    def get_classes(self) -> dict[LabelType, list[str]]:
        """Gets classes according to computer vision task.

        @rtype: dict[LabelType, list[str]]
        @return: A dictionary mapping tasks to their classes.
        """
        pass

    def get_skeletons(self) -> dict[str, dict] | None:
        """Returns the dictionary defining the semantic skeleton for each class using
        keypoints.

        @rtype: Dict[str, Dict]
        @return: A dictionary mapping classes to their skeleton definitions.
        """
        return None


def collate_fn(
    batch: list[LuxonisLoaderTorchOutput],
) -> tuple[Tensor, Labels]:
    """Default collate function used for training.

    @type batch: list[LuxonisLoaderTorchOutput]
    @param batch: List of images and their annotations in the LuxonisLoaderTorchOutput
        format.
    @rtype: tuple[Tensor, dict[LabelType, Tensor]]
    @return: Tuple of images and annotations in the format expected by the model.
    """
    imgs: tuple[Tensor, ...]
    labels: tuple[Labels, ...]
    imgs, labels = zip(*batch)

    out_labels = {}

    for task in labels[0].keys():
        label_type = labels[0][task][1]
        annos = [label[task][0] for label in labels]
        if label_type in [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]:
            out_labels[task] = torch.stack(annos, 0), label_type

        elif label_type in [LabelType.KEYPOINTS, LabelType.BOUNDINGBOX]:
            label_box: list[Tensor] = []
            for i, box in enumerate(annos):
                l_box = torch.zeros((box.shape[0], box.shape[1] + 1))
                l_box[:, 0] = i  # add target image index for build_targets()
                l_box[:, 1:] = box
                label_box.append(l_box)
            out_labels[task] = torch.cat(label_box, 0), label_type

    # exit()
    return torch.stack(imgs, 0), out_labels
