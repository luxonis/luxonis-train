from typing import Literal

from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks


class SigmoidFocalLoss(BaseLoss):
    """Sigmoid focal loss for binary or multi-label predictions.

    Metadata:
        - Module type: loss
        - Registry name: ``SigmoidFocalLoss``
        - Task: SEGMENTATION, CLASSIFICATION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar or unreduced sigmoid focal loss, depending on
          ``reduction``

    Prediction format:
        ``predictions`` contains logits compatible with ``target``.

    Target format:
        ``target`` contains binary labels with the same broadcast-compatible
        shape as ``predictions``.

    Formula:
        Applies torchvision sigmoid focal loss from the RetinaNet focal loss
        formulation.

    Provenance:
        - Source: torchvision
        - License: Project license
        - Implementation notes: Thin wrapper around
          ``torchvision.ops.sigmoid_focal_loss``.

    """

    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        """Focal loss from `Focal Loss for Dense Object Detection <https://arxiv.org/abs/1708.02002>`_.

        Args:
            alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples
                or -1 for ignore. Defaults to ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs hard
                examples. Defaults to ``2.0``.
            reduction (``Literal["none", "mean", "sum"]``): Reduction type for loss. Defaults to
                ``"mean"``.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        return sigmoid_focal_loss(
            predictions,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
