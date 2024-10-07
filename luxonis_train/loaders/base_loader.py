from abc import ABC, abstractmethod

from luxonis_ml.data import Augmentations
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size
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
    """Base abstract loader class that enforces LuxonisLoaderTorchOutput
    output label structure."""

    def __init__(
        self,
        view: str | list[str],
        augmentations: Augmentations | None = None,
        image_source: str | None = None,
    ):
        self.view = view if isinstance(view, list) else [view]
        self.augmentations = augmentations
        self._image_source = image_source

    @property
    def image_source(self) -> str:
        """Name of the input image group.

        Example: C{"image"}

        @type: str
        """
        if self._image_source is None:
            raise ValueError("image_source is not set")
        return self._image_source

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
