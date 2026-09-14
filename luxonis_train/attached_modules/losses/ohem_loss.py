"""A loss wrapper that averages only the hardest elements."""

from typing import Literal

import torch
from torch import Tensor

from luxonis_train.registry import LOSSES
from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss


class OHEMLoss(BaseLoss):
    r"""Online hard example mining wrapper for pixel or class losses.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, C, ...]`` logits, passed to
          the criterion
        - ``target`` (``Tensor``): the label of the task, passed to the
          criterion

    Outputs:
        - ``Tensor``: scalar

    Formula:
        The criterion computes the loss of each element, with
        ``reduction="none"``. The loss sorts these :math:`n` values in
        descending order, as :math:`\ell_0 \geq \ell_1 \geq \dots`. With
        the ratio :math:`r` = ``ohem_ratio`` and the probability
        :math:`p` = ``ohem_threshold``:

        .. math::

            k = \min\left(\lfloor r \, n \rfloor, n - 1\right),
            \qquad
            \tau = -\ln p

        When :math:`\ell_k > \tau`, the loss is the mean of all values
        above :math:`\tau`. Otherwise, the loss is the mean of the
        :math:`k` largest values. For a cross entropy without class
        weights and label smoothing, a value above :math:`\tau` means a
        predicted probability of the target below :math:`p`.

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        ``criterion="auto"`` selects `BCEWithLogitsLoss` for a node with
        one class, and `CrossEntropyLoss` otherwise. The loss returns
        ``nan`` when :math:`k = 0` and :math:`\ell_0 \leq \tau`, because
        the mean of no values is ``nan``. For an input without elements,
        it returns the empty tensor of the element losses.

    Example:
        Attached to a ``DDRNetSegmentationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: DDRNetSegmentationHead
              inputs: [DDRNet]
              losses:
                - name: OHEMLoss

    Compatible with:
        - Used by: `SegmentationModel`
        - Nodes:

          - `BiSeNetHead`
          - `ClassificationHead`
          - `DDRNetSegmentationHead`
          - `SegmentationHead`
          - `TransformerClassificationHead`
          - `TransformerSegmentationHead`

    """

    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        criterion: str | type[BaseLoss] | Literal["auto"] = "auto",
        ohem_ratio: float = 0.1,
        ohem_threshold: float = 0.7,
        **kwargs,
    ):
        r"""Initialize the loss and create the criterion.

        The method creates the criterion with ``**kwargs`` and
        ``reduction="none"``, so the criterion must accept
        ``reduction``.

        Args:
            criterion (``str | type[BaseLoss] | Literal["auto"]``): The
                loss that computes the value of each element. A string
                names a loss in the `LOSSES` registry, such as
                ``"CrossEntropyLoss"``. An unknown name raises
                ``KeyError``. The method uses a `BaseLoss` subclass as it
                is. ``"auto"`` selects ``"BCEWithLogitsLoss"`` when the
                node has one class, and ``"CrossEntropyLoss"`` otherwise.
                The method then logs a warning about the inferred task.
                Without a node, ``"auto"`` raises ``ValueError``.
            ohem_ratio (float): The ratio :math:`r` that sets the
                number :math:`k` of the largest values that the loss
                keeps when :math:`\ell_k \leq \tau`.
            ohem_threshold (float): The probability :math:`p` that sets
                the loss threshold :math:`\tau = -\ln p`.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseLoss` and to the criterion, such as
                ``final_loss_weight`` and ``node``. Both receive the same
                arguments, so an argument that only the criterion
                accepts, such as ``label_smoothing``, raises
                ``TypeError``.

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
        self._ohem_ratio = ohem_ratio
        self._ohem_threshold = -torch.log(torch.tensor(ohem_threshold))

        self._was_logged = False

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        r"""Compute the criterion loss and average the hardest elements.

        The criterion gives the loss of each element. The method flattens
        these losses and keeps the hardest ones, as the class formula
        describes.

        Args:
            predictions (``Tensor``): The main output of the node, passed
                to the criterion. For the default criteria, logits of
                shape ``[B, C, ...]``.
            target (``Tensor``): The single label of the task, such as
                the ``segmentation`` label of shape ``[B, C, H, W]``,
                passed to the criterion.

        Returns:
            ``Tensor``: The mean of the kept element losses, as a
            scalar. The value is ``nan`` when the method keeps no
            element. For an input without elements, the method returns
            the empty tensor of the element losses.

        Example:
            The four elements have the cross entropy ``0.1269`` or
            ``2.1269``. With ``ohem_ratio=0.5``, :math:`k` is ``2`` and
            :math:`\ell_2` is ``0.1269``. The default threshold gives
            :math:`\tau = -\ln 0.7 \approx 0.357`, so the loss keeps
            the two largest values. The threshold ``0.9`` gives
            :math:`\tau \approx 0.105`, so the loss keeps all values
            above it:

            >>> import torch
            >>> logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]]).repeat(2, 1)
            >>> target = torch.tensor([0, 0, 1, 1])
            >>> loss = OHEMLoss("CrossEntropyLoss", ohem_ratio=0.5)
            >>> round(loss(logits, target).item(), 4)
            2.1269
            >>> loss = OHEMLoss(
            ...     "CrossEntropyLoss", ohem_ratio=0.5, ohem_threshold=0.9
            ... )
            >>> round(loss(logits, target).item(), 4)
            1.1269

        """
        loss = self.criterion(predictions, target)
        assert isinstance(loss, Tensor)
        loss = loss.view(-1)

        n_pixels = loss.numel()

        if n_pixels == 0:
            return loss

        ohem_num = int(n_pixels * self._ohem_ratio)
        ohem_num = min(ohem_num, n_pixels - 1)

        loss, _ = loss.sort(descending=True)
        if loss[ohem_num] > self._ohem_threshold:
            loss = loss[loss > self._ohem_threshold]
        else:
            loss = loss[:ohem_num]

        return loss.mean()
