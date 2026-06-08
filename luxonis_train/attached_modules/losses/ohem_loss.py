from typing import Literal

import torch
from torch import Tensor

from luxonis_train.registry import LOSSES
from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss


class OHEMLoss(BaseLoss):
    """Online hard example mining wrapper for pixel or class losses.

    Metadata:
        - Module type: loss
        - Registry name: ``OHEMLoss``
        - Task: SEGMENTATION, CLASSIFICATION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar OHEM-filtered loss

    Prediction format:
        ``predictions`` is forwarded to the configured criterion.

    Target format:
        ``target`` is forwarded to the configured criterion.

    Formula:
        Computes unreduced criterion losses, sorts them descending, then averages
        losses above the threshold or the top ``ohem_ratio`` fraction.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Can infer ``BCEWithLogitsLoss`` or
          ``CrossEntropyLoss`` when ``criterion="auto"``.

    """

    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        criterion: str | type[BaseLoss] | Literal["auto"] = "auto",
        ohem_ratio: float = 0.1,
        ohem_threshold: float = 0.7,
        **kwargs,
    ):
        """Initialize the OHEM loss.

        Args:
            criterion (``BaseLoss | str | Literal["auto"]``): The criterion to use. It can be a string
                name of the criterion (e.g., "CrossEntropyLoss"), a class that inherits from
                ``BaseLoss``, or "auto" to infer the criterion based on the task and other
                parameters.
            ohem_ratio (float): The ratio of pixels to keep.
            ohem_threshold (float): The threshold for pixels to keep.
            **kwargs (``Any``): Additional keyword arguments that are passed to the criterion.

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
