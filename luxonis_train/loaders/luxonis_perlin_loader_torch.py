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
    @override
    def __init__(
        self,
        *args,
        anomaly_source_path: PathType,
        noise_prob: float = 0.5,
        beta: float | None = None,
        **kwargs,
    ):
        """Custom loader for LDF that adds Perlin noise during training
        with a given probability.

        @type anomaly_source_path: str
        @param anomaly_source_path: Path to the anomaly dataset from
            where random samples are drawn for noise.
        @type noise_prob: float
        @param noise_prob: The probability with which to apply Perlin
            noise.
        @type beta: float
        @param beta: The opacity of the anomaly mask. If None, a random
            value is chosen. It's advisable to set it to None.
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
        self.augmentations = self.loader.augmentations

    @override
    def get(self, idx: int) -> tuple[Tensor, Labels]:
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
