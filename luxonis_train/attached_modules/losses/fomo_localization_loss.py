import torch
import torch.nn.functional as F
from torch import Tensor

from luxonis_train.nodes import FOMOHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import get_center_keypoints

from .base_loss import BaseLoss


class FOMOLocalizationLoss(BaseLoss):
    """FOMO localization loss over heatmap logits.

    Metadata:
        - Module type: loss
        - Registry name: ``FOMOLocalizationLoss``
        - Task: FOMO
        - Attached node types: ``FOMOHead``
        - Inputs: ``heatmap``, ``target``
        - Outputs: scalar weighted focal BCE loss

    Prediction format:
        ``heatmap`` contains per-class logits shaped ``[B, C, H, W]``.

    Target format:
        ``target`` contains batch-indexed bounding boxes from which object
        centers are converted into heatmap keypoints.

    Formula:
        Builds a sparse target heatmap, applies BCE with logits, focal weighting,
        and an additional object-pixel weight.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Uses bounding-box centers as FOMO localization
          targets.

    """

    node: FOMOHead
    supported_tasks = [Tasks.FOMO]

    def __init__(
        self,
        object_weight: float = 500,
        alpha: float = 0.45,
        gamma: float = 2,
        **kwargs,
    ):
        """FOMO Localization Loss for object detection using heatmaps.

        Args:
            object_weight (float): Weight multiplier for keypoint pixels in loss
                calculation. Typical values range from 100-1000 depending on
                keypoint sparsity.
            alpha (float): Focal loss alpha parameter for class balance (0-1
                range). Lower values reduce positive example weighting.
            gamma (float): Focal loss gamma parameter for hard example focusing
                (gamma >= 0). Higher values focus more on hard misclassified
                examples.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)
        self.original_img_size = self.original_in_shape[1:]
        self.object_weight = object_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, heatmap: Tensor, target: Tensor) -> Tensor:
        batch_size, n_classes, height, width = heatmap.shape
        target_keypoints = get_center_keypoints(
            target, height=height, width=width
        ).int()
        target_heatmap = torch.zeros(
            (batch_size, n_classes, height, width), device=heatmap.device
        )

        for bbox, kpt in zip(target, target_keypoints, strict=True):
            class_id = int(bbox[1])
            batch_index, x_c, y_c = kpt[:3]
            target_heatmap[batch_index, class_id, y_c, x_c] = 1.0

        weight_matrix = torch.ones_like(target_heatmap)
        weight_matrix[target_heatmap == 1] = self.object_weight
        bce = F.binary_cross_entropy_with_logits(
            heatmap, target_heatmap, reduction="none"
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma

        weighted_loss = focal * bce * weight_matrix
        return weighted_loss.mean()
