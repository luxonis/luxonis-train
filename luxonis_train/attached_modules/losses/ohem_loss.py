from typing import Literal

import torch
from torch import Tensor

from luxonis_train.registry import LOSSES
from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss


class OHEMLoss(BaseLoss):
    """Generic OHEM loss that can be used with different criterions."""

    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        criterion: str | type[BaseLoss] | Literal["auto"] = "auto",
        ohem_ratio: float = 0.1,
        ohem_threshold: float = 0.7,
        **kwargs,
    ):
        """
    Args:
            criterion (BaseLoss | str | Literal["auto"]): The criterion to use. It can be a string
                name of the criterion (e.g., "CrossEntropyLoss"), a class that inherits from
                ``BaseLoss``, or "auto" to infer the criterion based on the task and other
                parameters.
            ohem_ratio (float): The ratio of pixels to keep.
            ohem_threshold (float): The threshold for pixels to keep.
            **kwargs (Any): Additional keyword arguments that are passed to the criterion.
        """
        super().__init__(**kwargs)

        if criterion == "auto":
            task = self._infer_torchmetrics_task(**kwargs)
            if task == "binary":
                criterion = "BCEWithLogitsLoss"
            else:
                criterion = "CrossEntropyLoss"

        if isinstance(criterion, str):
            criterion = LOSSES.get(criterion)

        self.criterion = criterion(**kwargs, reduction="none")
        self.ohem_ratio = ohem_ratio
        self.ohem_threshold = -torch.log(torch.tensor(ohem_threshold))

        self._was_logged = False

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        loss = self.criterion(predictions, target)
        assert isinstance(loss, Tensor)
        loss = loss.view(-1)

        n_pixels = loss.numel()

        if n_pixels == 0:
            return loss

        ohem_num = int(n_pixels * self.ohem_ratio)
        ohem_num = min(ohem_num, n_pixels - 1)

        loss, _ = loss.sort(descending=True)
        if loss[ohem_num] > self.ohem_threshold:
            loss = loss[loss > self.ohem_threshold]
        else:
            loss = loss[:ohem_num]

        return loss.mean()
