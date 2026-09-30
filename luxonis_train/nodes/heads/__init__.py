"""Heads that turn features into predictions.

Each head sets a task in its ``task`` class attribute. The task decides
which losses, metrics, and visualizers can attach to the head. The
``parser`` class attribute names the parser that reads the outputs of
the head in the exported model. A head without an export parser keeps
the empty default ``""``.

The heads, grouped by task:

- Classification: `ClassificationHead` and
  `TransformerClassificationHead`.
- Segmentation: `SegmentationHead`, `BiSeNetHead`,
  `DDRNetSegmentationHead`, and `TransformerSegmentationHead`.
- Bounding boxes: `EfficientBBoxHead` and `PrecisionBBoxHead`.
- Instance keypoints: `EfficientKeypointBBoxHead`.
- Instance segmentation: `PrecisionSegmentBBoxHead`.
- FOMO object centers: `FOMOHead`.
- Anomaly detection: `DiscSubNetHead`.
- OCR: `OCRCTCHead`.
- Embeddings: `GhostFaceNetHead`.

The bounding box, instance keypoint, and instance segmentation heads
derive from `BaseDetectionHead`. They run non-maximum suppression
(NMS) only in evaluation mode. Training mode and export mode skip NMS.
The metrics run in evaluation mode. Thus ``conf_thres``,
``iou_thres``, and ``max_det`` change what a metric sees. The NN
Archive stores the same three values for the export parser.

"""

from .base_head import BaseHead
from .bisenet_head import BiSeNetHead
from .classification_head import ClassificationHead
from .ddrnet_segmentation_head import DDRNetSegmentationHead
from .discsubnet_head import DiscSubNetHead
from .efficient_bbox_head import EfficientBBoxHead
from .efficient_keypoint_bbox_head import EfficientKeypointBBoxHead
from .fomo_head import FOMOHead
from .ghostfacenet_head import GhostFaceNetHead
from .ocr_ctc_head import OCRCTCHead
from .precision_bbox_head import PrecisionBBoxHead
from .precision_seg_bbox_head import PrecisionSegmentBBoxHead
from .segmentation_head import SegmentationHead
from .transformer_classification_head import TransformerClassificationHead
from .transformer_segmentation_head import TransformerSegmentationHead

__all__ = [
    "BaseHead",
    "BiSeNetHead",
    "ClassificationHead",
    "DDRNetSegmentationHead",
    "DiscSubNetHead",
    "EfficientBBoxHead",
    "EfficientKeypointBBoxHead",
    "FOMOHead",
    "GhostFaceNetHead",
    "OCRCTCHead",
    "PrecisionBBoxHead",
    "PrecisionSegmentBBoxHead",
    "SegmentationHead",
    "TransformerClassificationHead",
    "TransformerSegmentationHead",
]
