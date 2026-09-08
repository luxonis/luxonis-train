from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
from luxonis_ml.data import Category
from luxonis_ml.data.utils import get_task_type, task_is_metadata
from luxonis_ml.utils import AutoRegisterMeta
from torch import Size, Tensor
from torch.utils.data import Dataset

from luxonis_train.config.config import AugmentationConfig
from luxonis_train.registry import LOADERS
from luxonis_train.typing import Labels
from luxonis_train.utils.general import get_attribute_check_none

LuxonisLoaderTorchOutput = tuple[dict[str, Tensor] | Tensor, Labels]

MIXED_INPUT_TYPES_ERROR = (
    "All samples in a batch must have the same input type. "
    "Got a mix of tensors and dictionaries."
)


class BaseLoaderTorch(
    Dataset[LuxonisLoaderTorchOutput],
    ABC,
    metaclass=AutoRegisterMeta,
    register=False,
    registry=LOADERS,
):
    """Base abstract loader class for `LuxonisLoaderTorchOutput`
    samples.
    """

    def __init__(
        self,
        view: list[str],
        height: int | None = None,
        width: int | None = None,
        augmentation_engine: str = "albumentations",
        augmentation_config: list[AugmentationConfig] | None = None,
        image_source: str = "image",
        keep_aspect_ratio: bool = True,
        color_space: Literal["RGB", "BGR", "GRAY"] = "RGB",
        seed: int | None = None,
    ):
        """Initialize the base loader.

        Args:
            view (list[str]): Splits that form the view. Usually contains a
                single split, such as ``["train"]`` or ``["test"]``. More
                complex datasets can use multi-split views, such as
                ``["train_synthetic", "train_real"]``.
            height (int | None): Height of the output image.
            width (int | None): Width of the output image.
            augmentation_engine (str): Name of the augmentation engine. This
                can be used to swap between augmentation engines or select
                predefined engines, such as ``AlbumentationsEngine``.
            augmentation_config (list[AugmentationConfig] | None): List of
                augmentation configurations. Each configuration contains a
                ``name`` and a ``params`` dictionary.
            image_source (str): Name of the image source. This is only
                relevant for datasets with multiple image sources, such as
                ``"left"`` and ``"right"``, and defines which source is used
                for visualizations.
            keep_aspect_ratio (bool): Whether to keep the output image aspect
                ratio after resizing.
            color_space (``Literal["RGB", "BGR", "GRAY"]``): Output image color
                space.
            seed (int | None): Random seed used for augmentations.

        """
        self._view = view
        self._image_source = image_source
        self._augmentation_engine = augmentation_engine
        self._augmentation_config = augmentation_config
        self._height = height
        self._width = width
        self._keep_aspect_ratio = keep_aspect_ratio
        self._color_space = color_space
        self._seed = seed

    @property
    def image_source(self) -> str:
        """Str: Name of the input image group."""
        return self._getter_check_none("image_source")

    @property
    def view(self) -> list[str]:
        """List[str]: Splits forming this dataset's view."""
        return self._view

    @property
    def augmentation_engine(self) -> str:
        """Str: Name of the augmentation engine."""
        return self._getter_check_none("augmentation_engine")

    @property
    def augmentation_config(self) -> list[AugmentationConfig]:
        """List[AugmentationConfig]: Augmentation configurations."""
        return self._getter_check_none("augmentation_config")

    @property
    def height(self) -> int:
        """Int: Height of the output image."""
        return self._getter_check_none("height")

    @property
    def width(self) -> int:
        """Int: Width of the output image."""
        return self._getter_check_none("width")

    @property
    def keep_aspect_ratio(self) -> bool:
        """Bool: Whether to keep the output image aspect ratio after
        resizing.
        """
        return self._getter_check_none("keep_aspect_ratio")

    @property
    def color_space(self) -> Literal["RGB", "BGR"]:
        """``Literal["RGB", "BGR"]``: Color space of the output
        image.
        """
        return self._getter_check_none("color_space")

    @property
    def seed(self) -> int | None:
        """Int | None: Random seed used for augmentations."""
        return self._seed

    @property
    @abstractmethod
    def input_shapes(self) -> dict[str, Size]:
        """``dict[str, Size]``: Shape ``(c, h, w)`` of each loader
        group.

        Shapes do not include the batch dimension.

        Examples:
            Single image input::

                {
                    "image": torch.Size([3, 224, 224]),
                }

            Image and segmentation input::

                {
                    "image": torch.Size([3, 224, 224]),
                    "segmentation": torch.Size([1, 224, 224]),
                }

            Left image, right image, and disparity input::

                {
                    "left": torch.Size([3, 224, 224]),
                    "right": torch.Size([3, 224, 224]),
                    "disparity": torch.Size([1, 224, 224]),
                }

            Image, keypoints, and point cloud input::

                {
                    "image": torch.Size([3, 224, 224]),
                    "keypoints": torch.Size([17, 2]),
                    "point_cloud": torch.Size([20000, 3]),
                }

        """
        ...

    @property
    def input_shape(self) -> Size:
        """``Size``: Shape ``(c, h, w)`` of the input tensor without
        batch dimension.
        """
        return self.input_shapes[self.image_source]

    def augment_test_image(self, img: dict[str, Tensor] | Tensor) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose interface "
            "for test-time augmentation. Implement "
            "`augment_test_image` method to expose this functionality."
        )

    @abstractmethod
    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        """Load a sample from the dataset.

        Args:
            idx (int): Sample index.

        Returns:
            LuxonisLoaderTorchOutput: Sample data in
            `LuxonisLoaderTorchOutput` format.

        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_classes(self) -> dict[str, dict[str, int]]:
        """Get classes according to computer vision task.

        Returns:
            dict[str, dict[str, int]]: ``Mapping`` of task names to class name and
            class ID mappings.

        """
        ...

    def get_n_keypoints(self) -> dict[str, int] | None:
        """Get semantic skeleton definitions for classes using
        keypoints.

        Returns:
            dict[str, int] | None: ``Mapping`` of task names to keypoint counts, or
            ``None`` when keypoints are not available.

        """
        return None

    def get_metadata_types(
        self,
    ) -> dict[str, type[int] | type[Category] | type[float] | type[str]]:
        return {}

    def get_categorical_encodings(self) -> dict[str, dict[str, int]]:
        return {}

    def dict_numpy_to_torch(
        self, numpy_dictionary: dict[str, np.ndarray]
    ) -> dict[str, Tensor]:
        """Convert a dictionary of NumPy arrays to torch tensors.

        Args:
            numpy_dictionary (``dict[str, np.ndarray]``): Dictionary of NumPy
                arrays.

        Returns:
            ``dict[str, Tensor]``: Dictionary of torch tensors.

        """
        torch_dictionary = {}

        for task, array in numpy_dictionary.items():
            if array.dtype.kind == "U":
                array = np.array([ord(c) for c in array[0]], dtype=np.int32)
            torch_dictionary[task] = torch.tensor(array, dtype=torch.float32)

        return torch_dictionary

    def read_image(self, path: str) -> npt.NDArray[np.uint8]:
        """Read an unnormalized image from a file as a NumPy array.

        Args:
            path (str): ``Path`` to the image file.

        Returns:
            ``np.ndarray[np.uint8]``: Image as a NumPy array.

        Raises:
            ValueError: If the image cannot be read.

        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to read image from '{path}'")
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.color_space == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cast(npt.NDArray[np.uint8], img)

    def _getter_check_none(
        self,
        attribute: Literal[
            "view",
            "image_source",
            "augmentation_engine",
            "augmentation_config",
            "height",
            "width",
            "keep_aspect_ratio",
            "color_space",
        ],
    ) -> Any:
        return get_attribute_check_none(self, attribute)

    @staticmethod
    def img_numpy_to_torch(img: np.ndarray) -> Tensor:
        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))
        return torch.tensor(img, dtype=torch.float32)

    def collate_fn(
        self,
        batch: list[LuxonisLoaderTorchOutput],
    ) -> tuple[dict[str, Tensor] | Tensor, Labels]:
        """Default collate function used for training.

        Args:
            batch (list[LuxonisLoaderTorchOutput]): Loader outputs containing
                input tensors and labels in `LuxonisLoaderTorchOutput` format.

        Returns:
            ``tuple[dict[str, Tensor], Labels]``: Inputs and annotations in the
            format expected by the model.

        """
        inputs: tuple[dict[str, Tensor], ...] | tuple[Tensor, ...]
        labels: tuple[Labels, ...]
        inputs, labels = zip(*batch, strict=True)

        out_inputs = self._collate_inputs(inputs)
        out_labels = {
            task: self._collate_annotations(
                task, [label[task] for label in labels]
            )
            for task in labels[0]
        }
        return out_inputs, out_labels

    @staticmethod
    def _collate_inputs(
        inputs: tuple[dict[str, Tensor], ...] | tuple[Tensor, ...],
    ) -> dict[str, Tensor] | Tensor:
        first = inputs[0]
        if not isinstance(first, dict):
            tensors = [item for item in inputs if isinstance(item, Tensor)]
            if len(tensors) != len(inputs):
                raise TypeError(MIXED_INPUT_TYPES_ERROR)
            return torch.stack(tensors, 0)
        input_dicts = [item for item in inputs if isinstance(item, dict)]
        if len(input_dicts) != len(inputs):
            raise TypeError(MIXED_INPUT_TYPES_ERROR)
        return {
            name: torch.stack([item[name] for item in input_dicts], 0)
            for name in first
        }

    @staticmethod
    def _collate_annotations(task: str, annotations: list[Tensor]) -> Tensor:
        task_type = get_task_type(task)
        if task_type in {"keypoints", "boundingbox"}:
            return BaseLoaderTorch._add_batch_indices(annotations)
        if task_type == "instance_segmentation":
            return torch.cat(annotations, 0)
        if task_type == "metadata/text":
            return BaseLoaderTorch._pad_text_annotations(annotations)
        if task_is_metadata(task):
            return torch.cat(annotations, 0)
        return torch.stack(annotations, 0)

    @staticmethod
    def _add_batch_indices(annotations: list[Tensor]) -> Tensor:
        indexed_annotations = []
        for index, annotation in enumerate(annotations):
            indexed = torch.zeros(
                (annotation.shape[0], annotation.shape[1] + 1)
            )
            indexed[:, 0] = index
            indexed[:, 1:] = annotation
            indexed_annotations.append(indexed)
        return torch.cat(indexed_annotations, 0)

    @staticmethod
    def _pad_text_annotations(annotations: list[Tensor]) -> Tensor:
        max_length = max(len(annotation) for annotation in annotations)
        padded = torch.zeros(len(annotations), max_length, dtype=torch.int32)
        for index, annotation in enumerate(annotations):
            padded[index, : len(annotation)] = annotation
        return padded
