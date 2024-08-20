from abc import ABC, abstractmethod

import torch
from luxonis_ml.data import Augmentations
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor
from torch.utils.data import Dataset

from luxonis_train.utils.registry import LOADERS
from luxonis_train.utils.types import Labels, LabelType

LuxonisLoaderTorchOutput = tuple[dict[str, Tensor], Labels]
"""LuxonisLoaderTorchOutput is a tuple of source tensors and corresponding labels."""


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
        splits: str | list[str],
        augmentations: Augmentations | None = None,
        image_source: str | None = None,
    ):
        self.splits = splits if isinstance(splits, list) else [splits]
        self.augmentations = augmentations
        self._image_source = image_source

    @property
    def image_source(self) -> str:
        """Name of the input image group.

        Example: 'image'
        """
        if self._image_source is None:
            raise ValueError("image_source is not set")
        return self._image_source

    @property
    @abstractmethod
    def input_shapes(self) -> dict[str, Size]:
        """
        Shape of each loader group (sub-element), WITHOUT batch dimension.
        Examples:

        1. Single image input::
            {
                'image': torch.Size([3, 224, 224]),
            }

        2. Image and segmentation input::
            {
                'image': torch.Size([3, 224, 224]),
                'segmentation': torch.Size([1, 224, 224]),
            }

        3. Left image, right image and disparity input::
            {
                'left': torch.Size([3, 224, 224]),
                'right': torch.Size([3, 224, 224]),
                'disparity': torch.Size([1, 224, 224]),
            }

        4. Image, keypoints, and point cloud input::
            {
                'image': torch.Size([3, 224, 224]),
                'keypoints': torch.Size([17, 2]),
                'point_cloud': torch.Size([20000, 3]),
            }

        @rtype: dict[str, Size]
        @return: A dictionary mapping group names to their shapes.
        """
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
    def get_classes(self) -> dict[str, list[str]]:
        """Gets classes according to computer vision task.

        @rtype: dict[LabelType, list[str]]
        @return: A dictionary mapping tasks to their classes.
        """
        pass

    def get_n_keypoints(self) -> dict[str, int] | None:
        """Returns the dictionary defining the semantic skeleton for each class using
        keypoints.

        @rtype: Dict[str, Dict]
        @return: A dictionary mapping classes to their skeleton definitions.
        """
        return None


def collate_fn(
    batch: list[LuxonisLoaderTorchOutput],
) -> tuple[dict[str, Tensor], Labels]:
    """Default collate function used for training.

    @type batch: list[LuxonisLoaderTorchOutput]
    @param batch: List of loader outputs (dict of Tensors) and labels (dict of Tensors)
        in the LuxonisLoaderTorchOutput format.
    @rtype: tuple[dict[str, Tensor], dict[LabelType, Tensor]]
    @return: Tuple of inputs and annotations in the format expected by the model.
    """
    inputs: tuple[dict[str, Tensor], ...]
    labels: tuple[Labels, ...]
    inputs, labels = zip(*batch)

    out_inputs = {k: torch.stack([i[k] for i in inputs], 0) for k in inputs[0].keys()}
    out_labels = {task: {} for task in labels[0].keys()}

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

    return out_inputs, out_labels
