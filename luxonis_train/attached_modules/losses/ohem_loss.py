from typing import Any

import torch
from torch import Tensor

from luxonis_train.enums import TaskType

from .base_loss import BaseLoss


class OHEMLoss(BaseLoss[Tensor, Tensor]):
    """Generic OHEM loss that can be used with different criterions."""

    supported_tasks: list[TaskType] = [
        TaskType.SEGMENTATION,
        TaskType.CLASSIFICATION,
    ]

    def __init__(
        self,
        criterion: BaseLoss,
        ohem_ratio: float = 0.1,
        ohem_threshold: float = 0.7,
        **kwargs: Any,
    ):
        """Initializes the criterion.

        @type criterion: BaseLoss
        @param criterion: The criterion to use.
        @type ohem_ratio: float
        @param ohem_ratio: The ratio of pixels to keep.
        @type ohem_threshold: float
        @param ohem_threshold: The threshold for pixels to keep.
        """
        super().__init__(**kwargs)
        kwargs.update(reduction="none")
        self.criterion = criterion(**kwargs)
        self.ohem_ratio = ohem_ratio
        self.ohem_threshold = -torch.log(torch.tensor(ohem_threshold))

        self._was_logged = False

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        loss = self.criterion(preds, target).view(-1)

        num_pixels = loss.numel()

        if num_pixels == 0:
            return loss

        ohem_num = int(num_pixels * self.ohem_ratio)
        ohem_num = min(ohem_num, num_pixels - 1)

        loss, _ = loss.sort(descending=True)
        if loss[ohem_num] > self.ohem_threshold:
            loss = loss[loss > self.ohem_threshold]
        else:
            loss = loss[:ohem_num]

        return loss.mean()
