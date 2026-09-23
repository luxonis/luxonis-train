"""Loaders that feed batches to the training loop.

`LuxonisLoaderTorch` is the default. It reads an existing
``LuxonisDataset``, or parses a supported directory into a new one.
`LuxonisLoaderPerlinNoise` adds synthetic anomalies inside a Perlin
noise mask, for the anomaly detection task. `DummyLoader` yields
samples of zeros, so a model can build when no dataset is available.

Labels:
    `BaseLoaderTorch.collate_fn` returns each label of a batch under
    its ``"<task_name>/<label>"`` key. ``B`` is the batch size, and
    ``N`` is the number of instances in the whole batch.

    - ``classification``: ``[B, C]``, the multi-hot class vector of
      each image.
    - ``segmentation``: ``[B, C, H, W]``, one mask channel for each
      class.
    - ``boundingbox``: ``[N, 6]``, the rows
      ``[batch_index, class, x, y, w, h]``.
    - ``keypoints``: ``[N, 3K + 1]``, a batch index and then
      ``(x, y, visibility)`` for each of the ``K`` keypoints.
    - ``instance_segmentation``: ``[N, H, W]``, one mask for each box,
      in the order of the boxes.
    - ``metadata/text``: ``[B, S]``, the character codes of the text of
      each image, padded with zeros to the length ``S`` of the longest
      text. The ``ocr`` task reads this label.
    - Other ``metadata/<name>`` labels: the values of all samples,
      joined along the first dimension. The result has the shape
      ``[B]`` when each image has one value, such as the
      ``metadata/id`` label of the ``embeddings`` task.

Writing a custom loader:
    Subclass `BaseLoaderTorch` and implement ``input_shapes``,
    ``__len__``, ``__getitem__``, and ``get_classes``. A loader with
    keypoint labels must also implement ``get_n_keypoints``. Override
    ``collate_fn`` when the labels need other merge rules. Put the name
    of the subclass in the ``loader.name`` field of a config.

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
