from typing import Literal

from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks


class SigmoidFocalLoss(BaseLoss):
    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        """Focal loss from U{Focal Loss for Dense Object Detection
        <https://arxiv.org/abs/1708.02002>}.

        @type alpha: float
        @param alpha: Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore.
            Defaults to C{0.25}.
        @type gamma: float
        @param gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
            Defaults to C{2.0}.
        @type reduction: Literal["none", "mean", "sum"]
        @param reduction: Reduction type for loss. Defaults to C{"mean"}.
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
