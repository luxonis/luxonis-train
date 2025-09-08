from abc import ABC, abstractmethod
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
import torch
from luxonis_ml.data import Category
from luxonis_ml.data.utils import get_task_type, task_is_metadata
from luxonis_ml.typing import ConfigItem
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor
from torch.utils.data import Dataset

from luxonis_train.registry import LOADERS
from luxonis_train.typing import Labels
from luxonis_train.utils.general import get_attribute_check_none

LuxonisLoaderTorchOutput = tuple[dict[str, Tensor], Labels]


class BaseLoaderTorch(
    Dataset[LuxonisLoaderTorchOutput],
    ABC,
    metaclass=AutoRegisterMeta,
    register=False,
    registry=LOADERS,
):
    def __init__(
        self,
        view: list[str],
        height: int | None = None,
        width: int | None = None,
        augmentation_engine: str = "albumentations",
        augmentation_config: list[ConfigItem] | None = None,
        image_source: str = "image",
        keep_aspect_ratio: bool = True,
        color_space: Literal["RGB", "BGR", "GRAY"] = "RGB",
        seed: int | None = None,
    ):
        """Base abstract loader class that enforces
        LuxonisLoaderTorchOutput output label structure.

        @type view: list[str]
        @param view: List of splits that form the view.
            Usually contains only one split, e.g. C{["train"]}
            or C{["test"]}. However, more complex datasets
            can make use of multi-split views, e.g. C{["train_synthetic",
            "train_real"]}.

        @type height: int
        @param height: Height of the output image.

        @type width: int
        @param width: Width of the output image.

        @type augmentation_engine: str
        @param augmentation_engine: Name of the augmentation engine. Can
            be used to enable swapping between different augmentation engines or making use of pre-defined engines, e.g. C{AlbumentationsEngine}.

        @type augmentation_config: list[ConfigItem] | None
        @param augmentation_config: List of augmentation configurations.
            Individual configurations are in the form of::

                class ConfigItem:
                    name: str
                    params: dict[str, JsonValue]

            Where C{name} is the name of the augmentation and C{params} is a
            dictionary of its parameters.

            Example::

                ConfigItem(
                    name="HorizontalFlip",
                    params={"p": 0.5},
                )

        @type image_source: str
        @param image_source: Name of the image source. Only relevant for
            datasets with multiple image sources, e.g. C{"left"} and C{"right"}. This parameter defines which of these sources is used for
            visualizations.

        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether to keep the aspect ratio of the output image after resizing.

        @type color_space: Literal["RGB", "BGR", "GRAY"]
        @param color_space: Color space of the output image.

        @type seed: Optional[int]
        @param seed: The random seed to use for the augmentations.
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
        """Name of the input image group.

        @type: str
        """
        return self._getter_check_none("image_source")

    @property
    def view(self) -> list[str]:
        """List of splits forming this dataset's view.

        @type: list[str]
        """
        return self._view

    @property
    def augmentation_engine(self) -> str:
        """Name of the augmentation engine.

        @type: str
        """
        return self._getter_check_none("augmentation_engine")

    @property
    def augmentation_config(self) -> list[ConfigItem]:
        """List of augmentation configurations.

        @type: list[ConfigItem]
        """
        return self._getter_check_none("augmentation_config")

    @property
    def height(self) -> int:
        """Height of the output image.

        @type: int
        """
        return self._getter_check_none("height")

    @property
    def width(self) -> int:
        """Width of the output image.

        @type: int
        """
        return self._getter_check_none("width")

    @property
    def keep_aspect_ratio(self) -> bool:
        """Whether to keep the aspect ratio of the output image after
        resizing.

        @type: bool
        """
        return self._getter_check_none("keep_aspect_ratio")

    @property
    def color_space(self) -> Literal["RGB", "BGR"]:
        """Color space of the output image.

        @type: Literal["RGB", "BGR"]
        """
        return self._getter_check_none("color_space")

    @property
    def seed(self) -> int | None:
        """The random seed to use for the augmentations.

        @type: int | None
        """
        return self._seed

    @property
    @abstractmethod
    def input_shapes(self) -> dict[str, Size]:
        """
        Shape (c, h, w) of each loader group (sub-element), WITHOUT batch dimension.
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

        @type: dict[str, Size]
        """
        ...

    @property
    def input_shape(self) -> Size:
        """Shape (c, h, w) of the input tensor, WITHOUT batch dimension.

        @type: torch.Size
        """
        return self.input_shapes[self.image_source]

    def augment_test_image(self, img: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose interface "
            "for test-time augmentation. Implement "
            "`augment_test_image` method to expose this functionality."
        )

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        img, labels = self.get(idx)
        if isinstance(img, Tensor):
            img = {self.image_source: img}
        return img, labels

    @abstractmethod
    def __len__(self) -> int:
        """Returns length of the dataset."""
        ...

    @abstractmethod
    def get(self, idx: int) -> tuple[Tensor | dict[str, Tensor], Labels]:
        """Loads sample from dataset.

        @type idx: int
        @param idx: Sample index.
        @rtype: L{LuxonisLoaderTorchOutput}
        @return: Sample's data in L{LuxonisLoaderTorchOutput} format.
        """
        ...

    @abstractmethod
    def get_classes(self) -> dict[str, dict[str, int]]:
        """Gets classes according to computer vision task.

        @rtype: dict[LabelType, dict[str, int]]
        @return: A dictionary mapping tasks to their classes as a
            mappings from class name to class IDs.
        """
        ...

    def get_n_keypoints(self) -> dict[str, int] | None:
        """Returns the dictionary defining the semantic skeleton for
        each class using keypoints.

        @rtype: dict[str, Dict] | None
        @return: A dictionary mapping classes to their skeleton
            definitions.
        """
        return None

    def get_metadata_types(
        self,
    ) -> dict[str, type[int] | type[Category] | type[float] | type[str]]:
        return {}

    def dict_numpy_to_torch(
        self, numpy_dictionary: dict[str, np.ndarray]
    ) -> dict[str, Tensor]:
        """Converts a dictionary of numpy arrays to a dictionary of
        torch tensors.

        @type numpy_dictionary: dict[str, np.ndarray]
        @param numpy_dictionary: Dictionary of numpy arrays.
        @rtype: dict[str, Tensor]
        @return: Dictionary of torch tensors.
        """
        torch_dictionary = {}

        for task, array in numpy_dictionary.items():
            if array.dtype.kind in "U":
                array = np.array([ord(c) for c in array[0]], dtype=np.int32)
            torch_dictionary[task] = torch.tensor(array, dtype=torch.float32)

        return torch_dictionary

    def read_image(self, path: str) -> npt.NDArray[np.uint8]:
        """Reads an image from a file and returns an unnormalized image
        as a numpy array.

        @type path: str
        @param path: Path to the image file.
        @rtype: np.ndarray[np.uint8]
        @return: Image as a numpy array.
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.color_space == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

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
    ) -> tuple[dict[str, Tensor], Labels]:
        """Default collate function used for training.

        @type batch: list[LuxonisLoaderTorchOutput]
        @param batch: List of loader outputs (dict of Tensors) and
            labels (dict of Tensors) in the LuxonisLoaderTorchOutput
            format.
        @rtype: tuple[dict[str, Tensor], dict[str, Tensor]]
        @return: Tuple of inputs and annotations in the format expected
            by the model.
        """
        inputs: tuple[dict[str, Tensor], ...]
        labels: tuple[Labels, ...]
        inputs, labels = zip(*batch, strict=True)

        out_inputs = {
            k: torch.stack([i[k] for i in inputs], 0) for k in inputs[0]
        }

        out_labels: Labels = {}

        for task in labels[0]:
            task_type = get_task_type(task)
            annos = [label[task] for label in labels]

            if task_type in {"keypoints", "boundingbox"}:
                label_box: list[Tensor] = []
                for i, ann in enumerate(annos):
                    new_ann = torch.zeros((ann.shape[0], ann.shape[1] + 1))
                    # add batch index to separate boxes from different images
                    new_ann[:, 0] = i
                    new_ann[:, 1:] = ann
                    label_box.append(new_ann)
                out_labels[task] = torch.cat(label_box, 0)
            elif task_type == "instance_segmentation":
                out_labels[task] = torch.cat(annos, 0)
            elif task_is_metadata(task):
                if task_type == "metadata/text":
                    max_len = max(len(anno) for anno in annos)
                    padded_annos = torch.zeros(
                        len(annos), max_len, dtype=torch.int32
                    )
                    for i, anno in enumerate(annos):
                        padded_annos[i, : len(anno)] = anno
                    out_labels[task] = padded_annos
                else:
                    out_labels[task] = torch.cat(annos, 0)
            else:
                out_labels[task] = torch.stack(annos, 0)

        return out_inputs, out_labels
