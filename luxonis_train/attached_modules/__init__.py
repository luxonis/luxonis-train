r"""Losses, metrics, and visualizers that attach to a node.

An attached module reads the predictions of one node together with the
matching labels. The task of the node fixes the format of both. A loss
can ask its head for extra outputs, so the shapes below are the ones a
metric and a visualizer always see.

Predictions:
    - ``Tasks.CLASSIFICATION`` (``Tensor``):
      :math:`\left[B\right]`, one class index for each image
    - ``Tasks.SEGMENTATION`` (``Tensor``):
      :math:`\left[B, C, H, W\right]`, one channel for each class
    - ``Tasks.EMBEDDINGS`` (``Tensor``):
      :math:`\left[B, F\right]`, one feature vector for each image
    - ``Tasks.OCR`` (``Tensor`` or ``list[Tensor]``):
      :math:`\left[B, S\right]`, encoded token sequences
    - ``Tasks.BOUNDINGBOX`` (``list[Tensor]``):
      :math:`\left[N, 6\right]` for each image,
      ``[x1, y1, x2, y2, score, class]``
    - ``Tasks.INSTANCE_KEYPOINTS`` (``list[Tensor]``):
      :math:`\left[N, K, 3\right]` for each image, ``(x, y, visibility)``
    - ``Tasks.INSTANCE_SEGMENTATION`` (``list[Tensor]``):
      :math:`\left[N, H, W\right]` for each image, one mask for each
      instance

For the label formats, see `luxonis_train.loaders`.

"""

from .base_attached_module import BaseAttachedModule  # noqa: F401
from .losses import *
from .metrics import *
from .visualizers import *
