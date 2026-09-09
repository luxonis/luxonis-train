r"""Loaders that feed batches to the training loop.

`LuxonisLoaderTorch` is the default. It reads an existing
``LuxonisDataset``, or parses a supported directory into a new one.
`LuxonisLoaderPerlinNoise` adds Perlin noise to train the anomaly
detection model. `DummyLoader` produces empty batches, which lets a
config load with no data present.

Labels:
    ``collate_fn`` returns the batch in these shapes.

    - ``classification`` (``Tensor``):
      :math:`\left[B\right]`, one class index for each image
    - ``segmentation`` (``Tensor``):
      :math:`\left[B, C, H, W\right]`, one channel for each class
    - ``embeddings`` (``Tensor``):
      :math:`\left[B, K\right]`, one embedding for each image
    - ``ocr`` (``Tensor``):
      :math:`\left[B, S\right]`, token sequences padded with zeros
    - ``bboxes`` (``Tensor``):
      :math:`\left[N, 6\right]` over the whole batch,
      ``[batch, class, x, y, w, h]``
    - ``keypoints`` (``Tensor``):
      :math:`\left[N, 3K + 1\right]` over the whole batch, a batch
      index and then ``(x, y, visibility)`` for each keypoint
    - ``instance_segmentation`` (``Tensor``):
      :math:`\left[N, H, W\right]` over the whole batch, in the order
      of the boxes

Writing a custom loader:
    Subclass `BaseLoaderTorch` and implement ``input_shapes``,
    ``__len__``, ``__getitem__``, ``get_classes``, and ``collate_fn``.
    A loader that yields keypoints must also implement
    ``get_n_keypoints``. Name the subclass in the ``loader`` section of
    a config.

"""

from .base_loader import BaseLoaderTorch, LuxonisLoaderTorchOutput
from .dummy_loader import DummyLoader
from .luxonis_loader_torch import LuxonisLoaderTorch
from .luxonis_perlin_loader_torch import LuxonisLoaderPerlinNoise

__all__ = [
    "BaseLoaderTorch",
    "DummyLoader",
    "LuxonisLoaderPerlinNoise",
    "LuxonisLoaderTorch",
    "LuxonisLoaderTorchOutput",
]
