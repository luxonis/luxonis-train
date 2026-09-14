r"""Losses, metrics, and visualizers that attach to a node.

An attached module reads the output packet of one node and the labels
of the batch. `BaseLoss`, `BaseMetric`, and `BaseVisualizer` are the
base classes of the three kinds. `BaseAttachedModule` is the base class
of all three. A config lists the modules of a node under the
``losses``, ``metrics``, and ``visualizers`` keys of the node.

The parameter names of ``forward``, or of ``update`` for a metric,
select the inputs. For example, ``predictions`` gets the main
prediction of the task of the node, and ``target`` gets the only label
of the task. `BaseAttachedModule.get_parameters` gives all the rules.
For the label formats, see `luxonis_train.loaders`.

Predictions:
    This list gives the main prediction of each task in evaluation
    mode. It covers the tasks of the heads of this package. In training
    mode, the detection heads run no NMS. Their packets then do not
    hold the ``boundingbox``, ``keypoints``, or
    ``instance_segmentation`` key.

    - ``Tasks.CLASSIFICATION``: ``classification`` (``Tensor``),
      :math:`\left[B, n_{classes}\right]` logits
    - ``Tasks.SEGMENTATION``: ``segmentation`` (``Tensor``),
      :math:`\left[B, n_{classes}, H, W\right]` logits
    - ``Tasks.ANOMALY_DETECTION``: ``segmentation`` (``Tensor``),
      :math:`\left[B, C, H, W\right]` logits of the anomaly mask
    - ``Tasks.EMBEDDINGS``: ``embeddings`` (``Tensor``),
      :math:`\left[B, D\right]`, one embedding for each image
    - ``Tasks.OCR``: ``ocr`` (``Tensor``),
      :math:`\left[B, T, n_{classes}\right]` logits, one row for each
      step of the sequence
    - ``Tasks.BOUNDINGBOX``: ``boundingbox`` (``list[Tensor]``),
      :math:`\left[M_i, 6\right]` for each image,
      ``[x1, y1, x2, y2, score, class]`` in pixels
    - ``Tasks.INSTANCE_KEYPOINTS``: ``keypoints`` (``list[Tensor]``),
      :math:`\left[M_i, n_{keypoints}, 3\right]` for each image,
      ``(x, y, confidence)`` in pixels, next to ``boundingbox``
    - ``Tasks.INSTANCE_SEGMENTATION``: ``instance_segmentation``
      (``list[Tensor]``), :math:`\left[M_i, H, W\right]` binary masks
      for each image, next to ``boundingbox``
    - ``Tasks.FOMO``: ``heatmap`` (``Tensor``),
      :math:`\left[B, n_{classes}, H_f, W_f\right]` logits

"""

from .base_attached_module import BaseAttachedModule  # noqa: F401
from .losses import *
from .metrics import *
from .visualizers import *
