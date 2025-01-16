from abc import ABC, abstractmethod
from typing import Any, Literal

from luxonis_ml.typing import ConfigItem
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor
from torch.utils.data import Dataset

from luxonis_train.utils.registry import LOADERS

from .utils import LuxonisLoaderTorchOutput


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
        height: int,
        width: int,
        augmentation_engine: str = "albumentations",
        augmentation_config: list[ConfigItem] | None = None,
        image_source: str = "default",
        keep_aspect_ratio: bool = True,
        color_space: Literal["RGB", "BGR"] = "RGB",
    ):
        """Base abstract loader class that enforces
        LuxonisLoaderTorchOutput output label structure.

        @type view: list[str]
        @param view: List of view names. Usually contains only one element,
            e.g. C{["train"]} or C{["test"]}. However, more complex datasets
            can make use of multiple views, e.g. C{["train_synthetic",
            "train_real"]}

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
        @param image_source: Name of the input image group. This can be used for datasets with multiple image sources, e.g. left and right cameras or RGB and depth images. Irrelevant for datasets with only one image source.

        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether to keep the aspect ratio of the output image after resizing.

        @type color_space: Literal["RGB", "BGR"]
        @param color_space: Color space of the output image.
        """
        self._view = view
        self._image_source = image_source
        self._augmentation_engine = augmentation_engine
        self._augmentation_config = augmentation_config
        self._height = height
        self._width = width
        self._keep_aspect_ratio = keep_aspect_ratio
        self._color_space = color_space

    @property
    def image_source(self) -> str:
        """Name of the input image group.

        @type: str
        """
        return self._getter_check_none("image_source")

    @property
    def view(self) -> list[str]:
        """List of view names.

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
    def color_space(self) -> str:
        """Color space of the output image.

        @type: str
        """
        return self._getter_check_none("color_space")

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

    def augment_test_image(self, img: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose interface "
            "for test-time augmentation. Implement "
            "`augment_test_image` method to expose this functionality."
        )

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
        ...

    def get_n_keypoints(self) -> dict[str, int] | None:
        """Returns the dictionary defining the semantic skeleton for
        each class using keypoints.

        @rtype: Dict[str, Dict]
        @return: A dictionary mapping classes to their skeleton
            definitions.
        """
        return None

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
        value = getattr(self, f"_{attribute}")
        if value is None:
            raise ValueError(f"{attribute} is not set")
        return value
