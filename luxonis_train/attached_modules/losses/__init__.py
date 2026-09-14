"""Loss functions, grouped by the task they supervise.

- classification and segmentation: `CrossEntropyLoss`,
  `BCEWithLogitsLoss`, `SmoothBCEWithLogitsLoss`, `SigmoidFocalLoss`,
  `SoftmaxFocalLoss`, and `OHEMLoss`
- bounding boxes: `AdaptiveDetectionLoss` and
  `PrecisionDFLDetectionLoss`
- instance keypoints: `EfficientKeypointBBoxLoss`
- FOMO object centers: `FOMOLocalizationLoss`
- instance segmentation: `PrecisionDFLSegmentationLoss`
- anomaly detection: `ReconstructionSegmentationLoss`
- OCR: `CTCLoss`
- embeddings: the `pytorch-metric-learning
  <https://kevinmusgrave.github.io/pytorch-metric-learning/losses/>`_
  losses, wrapped by `EmbeddingLossWrapper`

A config attaches a loss to a node in the ``losses`` list of the node.
The ``Compatible with`` section of each loss lists the nodes that it
accepts. `BaseLoss` describes how to write a new loss. The training
step raises ``ValueError`` when no node has a loss.

"""

from .adaptive_detection_loss import AdaptiveDetectionLoss
from .base_loss import BaseLoss
from .bce_with_logits import BCEWithLogitsLoss
from .cross_entropy import CrossEntropyLoss
from .ctc_loss import CTCLoss
from .efficient_keypoint_bbox_loss import EfficientKeypointBBoxLoss
from .embedding_losses import EmbeddingLossWrapper
from .fomo_localization_loss import FOMOLocalizationLoss
from .ohem_loss import OHEMLoss
from .precision_dfl_detection_loss import PrecisionDFLDetectionLoss
from .precision_dfl_segmentation_loss import PrecisionDFLSegmentationLoss
from .reconstruction_segmentation_loss import ReconstructionSegmentationLoss
from .sigmoid_focal_loss import SigmoidFocalLoss
from .smooth_bce_with_logits import SmoothBCEWithLogitsLoss
from .softmax_focal_loss import SoftmaxFocalLoss

__all__ = [
    "AdaptiveDetectionLoss",
    "BCEWithLogitsLoss",
    "BaseLoss",
    "CTCLoss",
    "CrossEntropyLoss",
    "EfficientKeypointBBoxLoss",
    "EmbeddingLossWrapper",
    "FOMOLocalizationLoss",
    "OHEMLoss",
    "PrecisionDFLDetectionLoss",
    "PrecisionDFLSegmentationLoss",
    "ReconstructionSegmentationLoss",
    "SigmoidFocalLoss",
    "SmoothBCEWithLogitsLoss",
    "SoftmaxFocalLoss",
]
