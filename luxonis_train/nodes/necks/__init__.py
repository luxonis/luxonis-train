"""Necks that refine the feature maps of a backbone for a head.

- `RepPANNeck` fuses the feature maps of several scales. The detection,
  instance segmentation, and keypoint detection models use it.
- `SVTRNeck` refines the last feature map with SVTR transformer blocks.
  The OCR recognition model uses it.

"""

from .reppan_neck import RepPANNeck
from .svtr_neck import SVTRNeck

__all__ = ["RepPANNeck", "SVTRNeck"]
