from abc import ABC, abstractmethod

import torch
from luxonis_ml.data import Augmentations
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor
from torch.utils.data import Dataset

from luxonis_train.utils.registry import LOADERS
from luxonis_train.utils.types import Labels, LabelType

LuxonisLoaderTorchOutput = tuple[dict[str, Tensor], dict[str, Labels]]
"""LuxonisLoaderTorchOutput are two dictionaries, the first one contains the input data
and the second one contains the labels."""


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
        images_name: str | None = None,
    ):
        self.view = view
        self.augmentations = augmentations
        self._images_name = images_name

    @property
    def images_name(self) -> str:
        """Name of the input image group.

        Example: 'features'
        """
        return self._images_name

    @property
    @abstractmethod
    def input_shape(self) -> dict[str, Size]:
        """
        Shape of each loader group (sub-element), WITHOUT batch dimension.
        Examples:

        1. Single image input:
            {
                'image': torch.Size([3, 224, 224]),
            }

        2. Image and segmentation input:
            {
                'image': torch.Size([3, 224, 224]),
                'segmentation': torch.Size([1, 224, 224]),
            }

        3. Left image, right image and disparity input:
            {
                'left': torch.Size([3, 224, 224]),
                'right': torch.Size([3, 224, 224]),
                'disparity': torch.Size([1, 224, 224]),
            }

        4. Image, keypoints, and point cloud input:
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
) -> tuple[dict[str, Tensor], dict[str, dict[LabelType, Tensor]]]:
    """Default collate function used for training.

    @type batch: list[LuxonisLoaderTorchOutput]
    @param batch: List of loader outputs (dict of Tensors) and labels (dict of Tensors)
        in the LuxonisLoaderTorchOutput format.
    @rtype: tuple[dict[str, Tensor], dict[LabelType, Tensor]]
    @return: Tuple of inputs and annotations in the format expected by the model.
    """
    inputs, group_dicts = zip(*batch)

    # imgs = tuple[dict[str, Tensor]]. Stack the inputs into a single dict[str, Tensor].
    inputs = {k: torch.stack([i[k] for i in inputs], 0) for k in inputs[0].keys()}
    out_group_dicts = {task: {} for task in group_dicts[0].keys()}

    for task in list(group_dicts[0].keys()):
        anno_dicts = [group[task] for group in group_dicts]

        present_annotations = anno_dicts[0].keys()
        out_annotations: dict[LabelType, Tensor] = {
            anno: torch.empty(0) for anno in present_annotations
        }

        if LabelType.CLASSIFICATION in present_annotations:
            class_annos = [anno[LabelType.CLASSIFICATION] for anno in anno_dicts]
            out_annotations[LabelType.CLASSIFICATION] = torch.stack(class_annos, 0)

        if LabelType.SEGMENTATION in present_annotations:
            seg_annos = [anno[LabelType.SEGMENTATION] for anno in anno_dicts]
            out_annotations[LabelType.SEGMENTATION] = torch.stack(seg_annos, 0)

        if LabelType.BOUNDINGBOX in present_annotations:
            bbox_annos = [anno[LabelType.BOUNDINGBOX] for anno in anno_dicts]
            label_box: list[Tensor] = []
            for i, box in enumerate(bbox_annos):
                l_box = torch.zeros((box.shape[0], 6))
                l_box[:, 0] = i  # add target image index for build_targets()
                l_box[:, 1:] = box
                label_box.append(l_box)
            out_annotations[LabelType.BOUNDINGBOX] = torch.cat(label_box, 0)

        if LabelType.KEYPOINT in present_annotations:
            keypoint_annos = [anno[LabelType.KEYPOINT] for anno in anno_dicts]
            label_keypoints: list[Tensor] = []
            for i, points in enumerate(keypoint_annos):
                l_kps = torch.zeros((points.shape[0], points.shape[1] + 1))
                l_kps[:, 0] = i  # add target image index for build_targets()
                l_kps[:, 1:] = points
                label_keypoints.append(l_kps)
            out_annotations[LabelType.KEYPOINT] = torch.cat(label_keypoints, 0)

        out_group_dicts[task] = out_annotations

    return inputs, out_group_dicts
