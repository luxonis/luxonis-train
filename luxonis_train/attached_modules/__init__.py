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

Prediction packets use these shapes in evaluation:

- Classification: ``[B, n_classes]`` logits.
- Segmentation and anomaly detection: ``[B, C, H, W]`` logits.
- Embeddings: ``[B, D]``.
- OCR: ``[B, T, n_classes]`` logits.
- Bounding boxes: one ``[M_i, 6]`` tensor per image, with rows
  ``[x1, y1, x2, y2, score, class]`` in pixels.
- Keypoints: one ``[M_i, n_keypoints, 3]`` tensor per image, with
  ``(x, y, confidence)`` in pixels.
- Instance masks: one ``[M_i, H, W]`` tensor per image.
- FOMO: ``[B, n_classes, H_f, W_f]`` heatmap logits.

Detection heads skip non-maximum suppression during training, so their
packets do not yet contain the final boxes, keypoints, or instance
masks. The heads document their packet keys; `luxonis_train.loaders`
documents the matching label formats.

"""

from .base_attached_module import BaseAttachedModule  # noqa: F401
from .losses import *
from .metrics import *
from .visualizers import *
