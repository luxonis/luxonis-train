"""Necks that fuse the feature maps of a backbone.

`RepPANNeck` merges features across strides for the detection heads.
`SVTRNeck` refines a feature sequence for the OCR head.

"""

from .reppan_neck import RepPANNeck
from .svtr_neck import SVTRNeck

__all__ = ["RepPANNeck", "SVTRNeck"]
