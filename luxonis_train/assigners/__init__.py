"""Assigners that match the anchors of a detection head to the ground
truth boxes.

An assigner decides which anchors predict each ground truth box. It
then builds the per-anchor labels, boxes, and scores that the
detection losses read. The package has two assigners:

- `ATSSAssigner` selects the positive anchors from the anchor geometry
  and an adaptive IoU threshold.
- `TaskAlignedAssigner` selects the positive anchors by a metric that
  combines the predicted class score and the IoU of the predicted box.

"""

from .atss_assigner import ATSSAssigner
from .tal_assigner import TaskAlignedAssigner

__all__ = ["ATSSAssigner", "TaskAlignedAssigner"]
