"""The loader of the anomaly detection task.

`LuxonisLoaderPerlinNoise` blends a texture image into a clean image,
inside a random Perlin noise mask. The mask is the label.

"""

import random
from collections.abc import Generator, Mapping
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from bidict import bidict
from luxonis_ml.typing import PathType
from luxonis_ml.utils import LuxonisFileSystem
from torch import Tensor
from typing_extensions import override

from luxonis_train.typing import Labels

from .luxonis_loader_torch import LuxonisLoaderTorch
from .perlin import apply_anomaly_to_img


class LuxonisLoaderPerlinNoise(LuxonisLoaderTorch):
    """Loader that adds synthetic anomalies for the anomaly detection
    task.

    The dataset must have only one task. When the first split of the
    view is ``"train"``, the loader adds an anomaly to an image with the
    probability ``noise_prob``. The anomaly is a random texture image
    inside a random Perlin noise mask, see
    `luxonis_train.loaders.perlin.apply_anomaly_to_img`. The image
    height and width of the ``train`` view must then be multiples of
    ``32``. For other views, the loader reads the anomaly mask from the
    ``segmentation`` label of the dataset.

    Example:
        The ``loader`` section of a config:

        .. code-block:: yaml

            loader:
              name: LuxonisLoaderPerlinNoise
              params:
                dataset_name: mvtec_v2
                anomaly_source_path: ../data/dtd/images/

    """

    @override
    def __init__(
        self,
        *args,
        anomaly_source_path: PathType,
        noise_prob: float = 0.5,
        beta: float | None = None,
        **kwargs,
    ):
        """Initialize the dataset and collect the texture images.

        Args:
            *args (``Any``): Positional arguments for
                `LuxonisLoaderTorch`.
            anomaly_source_path (``PathType``): The directory of the
                texture images. The loader collects all files in the
                directory tree with an extension from ``IMAGE_FORMATS``,
                in any letter case. The loader downloads a remote URL
                into ``./data``, and uses a local path directly.
            noise_prob (float): The probability that a sample of the
                ``train`` view gets an anomaly.
            beta (float | None): The weight of the clean image inside the
                mask. The texture gets the weight ``1 - beta``, so
                ``0.0`` gives an opaque anomaly. ``None`` draws a new
                value from ``[0, 0.8)`` for each anomaly.
            **kwargs (``Any``): Keyword arguments for
                `LuxonisLoaderTorch`.

        Raises:
            FileNotFoundError: If the download of ``anomaly_source_path``
                fails, or the directory has no image files.
            ValueError: If the dataset has more than one task.

        """
        super().__init__(*args, **kwargs)

        if isinstance(anomaly_source_path, str):
            try:
                self.anomaly_source_path = LuxonisFileSystem.download(
                    anomaly_source_path, dest="./data"
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"The anomaly source path '{anomaly_source_path}' "
                    "could not be found or downloaded."
                ) from e
        else:
            self.anomaly_source_path = anomaly_source_path

        from luxonis_train.core.utils.infer_utils import IMAGE_FORMATS

        self.anomaly_files = [
            f
            for f in self.anomaly_source_path.rglob("*")
            if f.suffix.lower() in IMAGE_FORMATS
        ]
        if not self.anomaly_files:
            raise FileNotFoundError(
                "No image files found at the specified path."
            )

        self.noise_prob = noise_prob
        if len(self.loader.dataset.get_tasks()) > 1:
            # TODO: Can be extended to multiple tasks
            raise ValueError(
                "This loader only supports datasets with a single task."
            )
        self.beta = beta
        self.task_name = next(iter(self.loader.dataset.get_tasks()))
        self.augmentations = self.loader._augmentations

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Labels]:
        """Load a sample and build the anomaly labels.

        The method saves the Python and NumPy random states before it
        reads the sample, and restores them after the read. When the
        first split of the view is ``"train"``, the image gets
        an anomaly with the probability ``noise_prob``. The texture is a
        random file from ``anomaly_source_path``, with the augmentations
        of the loader. A train image without an anomaly gets an empty
        mask. For other views, the mask is the last channel of the
        ``segmentation`` label of the dataset. The method ignores
        ``return_sample_metadata`` and ``kpts_mapping_per_task``.

        Args:
            idx (int): The index of the sample.

        Returns:
            ``tuple[Tensor, Labels]``: The image of shape ``[C, H, W]``,
            with the anomaly when it has one. The labels have two keys,
            where ``task`` is the name of the dataset task:

            - ``"task/segmentation"``: The one-hot anomaly mask of shape
              ``[2, H, W]``. Channel ``1`` marks the anomaly.
            - ``"task/original_segmentation"``: The image before the
              anomaly, of shape ``[C, H, W]``.

            The labels do not include the other labels of the dataset.

        Raises:
            NotImplementedError: If the dataset has more than one image
                source and ``image_source`` is ``None``.

        """
        with _freeze_seed():
            img, labels = self.loader[idx]
        if isinstance(img, dict):
            if self._image_source is None:
                raise NotImplementedError(
                    "This loader does not support multi-input models "
                    "and the `image_source` identifying the input image "
                    "is not set. Please set `image_source` to a valid "
                    "image source in the loader parameters or use a dataset "
                    "with a single image source."
                )
            img = img[self._image_source]

        img = self.img_numpy_to_torch(img)
        tensor_labels = self.dict_numpy_to_torch(labels)

        if self.view[0] == "train":
            if random.random() < self.noise_prob:
                anomaly_path = random.choice(self.anomaly_files)
                anomaly_img = self.read_image(str(anomaly_path))

                if self.augmentations is not None:
                    anomaly_img = self.augmentations.apply(
                        [({self.image_source: anomaly_img}, {})]
                    )[0][self.image_source]

                anomaly_img = self.img_numpy_to_torch(anomaly_img)
                aug_tensor_img, an_mask = apply_anomaly_to_img(
                    img, anomaly_img, self.beta
                )
            else:
                aug_tensor_img = img
                an_mask = torch.zeros((self.height, self.width))
        else:
            aug_tensor_img = img
            an_mask = torch.tensor(
                labels.pop(f"{self.task_name}/segmentation")
            )[-1, ...]

        an_mask = F.one_hot(an_mask.long(), 2).permute(2, 0, 1).float()

        tensor_labels = {
            f"{self.task_name}/original_segmentation": img,
            f"{self.task_name}/segmentation": an_mask,
        }

        return aug_tensor_img, tensor_labels

    @override
    def get_classes(self) -> dict[str, Mapping[str, int]]:
        """Return the two classes of the anomaly mask.

        Returns:
            ``dict[str, Mapping[str, int]]``: One entry for the dataset
            task. Its ``bidict`` maps ``"background"`` to ``0`` and
            ``"anomaly"`` to ``1``.

        """
        names = ["background", "anomaly"]
        idx_map = bidict({name: i for i, name in enumerate(names)})
        return {self.task_name: idx_map}


@contextmanager
def _freeze_seed() -> Generator:
    python_seed = random.getstate()
    numpy_seed = np.random.get_state()
    yield
    random.setstate(python_seed)
    np.random.set_state(numpy_seed)
