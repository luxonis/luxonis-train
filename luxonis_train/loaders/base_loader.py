"""The base class every loader inherits, and the type of one sample.

`LuxonisLoaderTorchOutput` is the type of one sample. It is a pair of
the input and the labels. The input is an image of shape ``[C, H, W]``,
or a dictionary that maps each input name to its image. The labels map
each ``"<task_name>/<label>"`` key to a tensor.

"""

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
    """Base class for the loaders of the training pipeline.

    A subclass registers in `luxonis_train.registry.LOADERS` under its
    class name, so the ``loader.name`` field of a config can name it. A
    subclass must implement `input_shapes`, ``__len__``,
    ``__getitem__``, and `get_classes`. A loader with keypoint labels
    must also override `get_n_keypoints`.

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
        """Store the settings that every loader shares.

        Args:
            view (list[str]): The splits that form the view. The list
                usually holds one split, such as ``["train"]``. A dataset
                can also combine splits, such as
                ``["train_synthetic", "train_real"]``.
            height (int | None): The height of the output image. With
                ``None``, the `height` property raises ``ValueError``.
            width (int | None): The width of the output image. With
                ``None``, the `width` property raises ``ValueError``.
            augmentation_engine (str): The name of the augmentation
                engine, such as ``"albumentations"``.
            augmentation_config (list[AugmentationConfig] | None): The
                augmentations. Each item has a ``name`` and a ``params``
                dictionary. With ``None``, the `augmentation_config`
                property raises ``ValueError``.
            image_source (str): The name of the main image source. For a
                dataset with more than one source, such as ``"left"`` and
                ``"right"``, the visualizations use this source.
            keep_aspect_ratio (bool): Whether the resize keeps the aspect
                ratio of the image.
            color_space (``Literal["RGB", "BGR", "GRAY"]``): The color
                space of the output image.
            seed (int | None): The random seed of the augmentations, or
                ``None``.

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
        """The name of the main image source, such as ``"image"``."""
        return self._getter_check_none("image_source")

    @property
    def view(self) -> list[str]:
        """The splits that form the view, such as ``["train"]``."""
        return self._view

    @property
    def augmentation_engine(self) -> str:
        """The name of the augmentation engine."""
        return self._getter_check_none("augmentation_engine")

    @property
    def augmentation_config(self) -> list[AugmentationConfig]:
        """The augmentations of the loader.

        The property raises ``ValueError`` when the constructor got
        ``None``.

        """
        return self._getter_check_none("augmentation_config")

    @property
    def height(self) -> int:
        """The height of the output image.

        The property raises ``ValueError`` when the constructor got
        ``None``.

        """
        return self._getter_check_none("height")

    @property
    def width(self) -> int:
        """The width of the output image.

        The property raises ``ValueError`` when the constructor got
        ``None``.

        """
        return self._getter_check_none("width")

    @property
    def keep_aspect_ratio(self) -> bool:
        """Whether the resize keeps the aspect ratio of the image."""
        return self._getter_check_none("keep_aspect_ratio")

    @property
    def color_space(self) -> Literal["RGB", "BGR"]:
        """The color space of the output image.

        The value is ``"RGB"``, ``"BGR"``, or ``"GRAY"``.

        """
        return self._getter_check_none("color_space")

    @property
    def seed(self) -> int | None:
        """The random seed of the augmentations, or ``None``."""
        return self._seed

    @property
    @abstractmethod
    def input_shapes(self) -> dict[str, Size]:
        """The shape of each input of one sample, keyed by the input
        name.

        An implementation returns one shape for each input, without the
        batch dimension. An image has the shape ``[C, H, W]``. The
        result must hold the `image_source` key, because `input_shape`
        reads it.

        Examples:
            A loader with one image:

            .. code-block:: python

                {
                    "image": torch.Size([3, 224, 224]),
                }

            A loader with an image and a segmentation input:

            .. code-block:: python

                {
                    "image": torch.Size([3, 224, 224]),
                    "segmentation": torch.Size([1, 224, 224]),
                }

            A loader with a left image, a right image, and a disparity
            map:

            .. code-block:: python

                {
                    "left": torch.Size([3, 224, 224]),
                    "right": torch.Size([3, 224, 224]),
                    "disparity": torch.Size([1, 224, 224]),
                }

            A loader with an image, keypoints, and a point cloud:

            .. code-block:: python

                {
                    "image": torch.Size([3, 224, 224]),
                    "keypoints": torch.Size([17, 2]),
                    "point_cloud": torch.Size([20000, 3]),
                }

        """
        ...

    @property
    def input_shape(self) -> Size:
        """The shape ``[C, H, W]`` of the `image_source` input.

        The shape comes from `input_shapes` and has no batch dimension.

        """
        return self.input_shapes[self.image_source]

    def augment_test_image(self, img: dict[str, Tensor] | Tensor) -> Tensor:
        """Apply the augmentations of the loader to one raw image.

        Inference calls this method to prepare an image like the samples
        of the view. The base implementation only raises. A loader that
        supports inference on raw images must override it.
        `LuxonisLoaderTorch` overrides it.

        Args:
            img (``dict[str, Tensor] | Tensor``): The raw image of shape
                ``[H, W, C]``. A dictionary maps each source name to its
                image.

        Returns:
            ``Tensor``: An override returns the augmented image of shape
            ``[H, W, C]``.

        Raises:
            NotImplementedError: Always, in the base implementation.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose interface "
            "for test-time augmentation. Implement "
            "`augment_test_image` method to expose this functionality."
        )

    @abstractmethod
    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        """Load one sample of the view.

        Args:
            idx (int): The index of the sample.

        Returns:
            LuxonisLoaderTorchOutput: The input and the labels of the
            sample. The input is an image of shape ``[C, H, W]``, or a
            dictionary that maps each input name to its image. The
            labels map each ``"<task_name>/<label>"`` key to a tensor.

        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the view.

        Returns:
            int: The number of samples.

        """
        ...

    @abstractmethod
    def get_classes(self) -> dict[str, dict[str, int]]:
        """Return the class names and the class IDs of each task.

        Returns:
            dict[str, dict[str, int]]: The class name to class ID
            mapping of each task, keyed by the task name.

        """
        ...

    def get_n_keypoints(self) -> dict[str, int] | None:
        """Return the number of keypoints of each task.

        The base implementation returns ``None``. A loader with keypoint
        labels must override it.

        Returns:
            dict[str, int] | None: The number of keypoints, keyed by the
            task name, or ``None`` when the loader has no keypoints.

        """
        return None

    def get_metadata_types(
        self,
    ) -> dict[str, type[int] | type[Category] | type[float] | type[str]]:
        """Return the Python type of each metadata label.

        The base implementation returns an empty dictionary, which means
        that the loader has no metadata labels. `DatasetMetadata` reads
        the result.

        Returns:
            ``dict[str, type[int] | type[Category] | type[float] | type[str]]``:
            The type of each metadata label, keyed by the label name,
            such as ``"task_name/metadata/color"``.

        """
        return {}

    def get_categorical_encodings(self) -> dict[str, dict[str, int]]:
        """Return the integer code of each category of each metadata
        label.

        The base implementation returns an empty dictionary, which means
        that the loader has no categorical metadata labels.

        Returns:
            dict[str, dict[str, int]]: The category to code mapping of
            each categorical metadata label, keyed by the label name.

        """
        return {}

    def dict_numpy_to_torch(
        self, numpy_dictionary: dict[str, np.ndarray]
    ) -> dict[str, Tensor]:
        """Convert a dictionary of NumPy arrays to ``torch.float32``
        tensors.

        A string array becomes the character codes of its first string.
        The method converts the codes to ``torch.float32`` too.

        Args:
            numpy_dictionary (``dict[str, np.ndarray]``): The arrays,
                such as the labels of one sample.

        Returns:
            ``dict[str, Tensor]``: A new dictionary with the same keys and
            one tensor for each array.

        """
        torch_dictionary = {}

        for task, array in numpy_dictionary.items():
            if array.dtype.kind == "U":
                array = np.array([ord(c) for c in array[0]], dtype=np.int32)
            torch_dictionary[task] = torch.tensor(array, dtype=torch.float32)

        return torch_dictionary

    def read_image(self, path: str) -> npt.NDArray[np.uint8]:
        """Read an image file into an unnormalized NumPy array.

        OpenCV reads the file as a BGR color image. The method then
        converts it to `color_space`.

        Args:
            path (str): The path to the image file.

        Returns:
            ``npt.NDArray[np.uint8]``: The image of shape ``[H, W, 3]``,
            or ``[H, W]`` for the ``"GRAY"`` color space.

        Raises:
            ValueError: If OpenCV cannot read the file.

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
        """Convert a NumPy image to a ``torch.float32`` tensor.

        The method moves the channels of a 3D image first. A 2D image
        keeps its shape.

        Args:
            img (``np.ndarray``): The image of shape ``[H, W, C]`` or
                ``[H, W]``.

        Returns:
            ``Tensor``: The image of shape ``[C, H, W]`` or ``[H, W]``.

        Example:
            >>> import numpy as np
            >>> image = np.zeros((4, 6, 3), dtype=np.uint8)
            >>> BaseLoaderTorch.img_numpy_to_torch(image).shape
            torch.Size([3, 4, 6])
            >>> gray = np.zeros((4, 6), dtype=np.uint8)
            >>> BaseLoaderTorch.img_numpy_to_torch(gray).shape
            torch.Size([4, 6])

        """
        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))
        return torch.tensor(img, dtype=torch.float32)

    def collate_fn(
        self,
        batch: list[LuxonisLoaderTorchOutput],
    ) -> tuple[dict[str, Tensor] | Tensor, Labels]:
        """Merge a list of samples into one batch.

        The method stacks the inputs along a new first dimension. For
        dictionary inputs, it stacks each input name separately. It
        merges each label of the first sample by the label type:

        - ``boundingbox`` and ``keypoints``: The method adds the index
          of the sample as a new first column, and joins the rows of all
          samples. A ``[N, 5]`` box label becomes ``[N, 6]``.
        - ``instance_segmentation``: The method joins the masks of all
          samples along the first dimension.
        - ``metadata/text``: The method pads the character codes of
          each sample with zeros to the longest text. The result is a
          ``torch.int32`` tensor of shape ``[B, S]``.
        - Other ``metadata/<name>`` labels: The method joins the values
          of all samples along the first dimension.
        - Other labels, such as ``classification`` and
          ``segmentation``: The method stacks them along a new first
          dimension.

        Args:
            batch (list[LuxonisLoaderTorchOutput]): The samples. All
                samples must have the label keys of the first sample.

        Returns:
            ``tuple[dict[str, Tensor] | Tensor, Labels]``: The batched
            input and the batched labels, with the keys of the first
            sample.

        Raises:
            TypeError: If the batch mixes tensor inputs and dictionary
                inputs.

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
