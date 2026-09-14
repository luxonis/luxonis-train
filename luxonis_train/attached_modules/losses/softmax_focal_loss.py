"""Softmax focal loss, which lowers the weight of the easy examples."""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, amp

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks


class SoftmaxFocalLoss(BaseLoss):
    r"""Focal loss on softmax outputs, for multiclass predictions.

    The loss applies a softmax over the class dimension. The focal
    factor lowers the loss of the elements whose target class already
    has a high probability.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, C, ...]`` logits, with
          :math:`C \geq 2`
        - ``targets`` (``Tensor``): same shape, one-hot

    Outputs:
        - ``Tensor``: scalar, or ``[B, ...]`` when ``reduction`` is
          ``"none"``

    Formula:
        At one element, :math:`p_c` is the softmax probability of the
        class :math:`c`, and :math:`y_c` is its target. For a smoothing
        factor :math:`s > 0`, the loss first clips each target to
        :math:`\left[s / (C - 1), 1 - s\right]`. The loss of the element
        is then

        .. math::

            p_t = \sum_c y_c \, p_c + s, \qquad
            \ell = -\alpha_t \, (1 - p_t)^{\gamma} \log p_t

        A float ``alpha`` is :math:`\alpha_t` for every element. A list
        ``alpha`` gives :math:`\alpha_t` from the entry of the target
        class, the ``argmax`` of the targets. ``reduction`` then takes
        the mean or the sum over all elements, or keeps the loss of each
        element.

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        The loss runs in ``float32`` with autocast turned off, also in a
        mixed precision run. With ``alpha=1``, ``gamma=0``, and
        ``smooth=0``, the loss is the cross entropy.

    Example:
        Attached to a ``DDRNetSegmentationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: DDRNetSegmentationHead
              inputs: [DDRNet]
              losses:
                - name: SoftmaxFocalLoss

    Compatible with:
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
        alpha: float | list[float] = 0.25,
        gamma: float = 2.0,
        smooth: float = 0.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        r"""Initialize the loss and check the smoothing factor.

        Args:
            alpha (float | list[float]): The class weight
                :math:`\alpha_t`. A float scales the loss of every
                element by the same factor, so it does not favor a
                class. A list holds one weight for each class, in class
                order. `forward` then checks that the list has one entry
                for each class.
            gamma (float): The exponent of the focal factor
                :math:`(1 - p_t)`. ``0`` turns the focal factor off.
            smooth (float): The label smoothing factor :math:`s`, in
                ``[0, 1]``. The class formula shows how it changes the
                targets and :math:`p_t`.
            reduction (``Literal["none", "mean", "sum"]``): How to
                reduce the loss of the elements:

                - ``"none"``: return the loss of each element.
                - ``"mean"``: return the mean over all elements.
                - ``"sum"``: return the sum over all elements.

                `forward` treats any other value as ``"none"``.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseLoss`, such as ``final_loss_weight`` and ``node``.

        Raises:
            ValueError: When ``smooth`` is outside ``[0, 1]``.

        """
        super().__init__(**kwargs)

        self._gamma = gamma
        self._smooth = smooth
        self._reduction = reduction

        if isinstance(alpha, list):
            self._alpha = torch.tensor(alpha)
        else:
            self._alpha = alpha

        if self._smooth is not None and not (0 <= self._smooth <= 1.0):
            raise ValueError("smooth value should be in [0,1]")

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the softmax focal loss between logits and targets.

        Args:
            predictions (``Tensor``): Logits of shape ``[B, C, ...]``,
                with at least two classes. The main output of the node.
            targets (``Tensor``): One-hot targets of the same shape as
                ``predictions``.

        Returns:
            ``Tensor``: A ``float32`` scalar for the ``"mean"`` and
            ``"sum"`` reductions. For any other ``reduction``, the loss
            of each element, of shape ``[B, ...]``.

        Raises:
            ValueError: When ``predictions`` has fewer than two classes,
                when the shapes differ, or when a list ``alpha`` does
                not have one entry for each class.

        Examples:
            With ``alpha=1`` and ``gamma=0``, the loss is the cross
            entropy:

            >>> import torch
            >>> import torch.nn.functional as F
            >>> logits = torch.tensor([[2.0, 0.0, -1.0]])
            >>> target = torch.tensor([[1.0, 0.0, 0.0]])
            >>> loss = SoftmaxFocalLoss(alpha=1.0, gamma=0.0)
            >>> ce = F.cross_entropy(logits, target)
            >>> torch.allclose(loss(logits, target), ce)
            True

            One class is not enough:

            >>> loss(torch.zeros(1, 1), torch.ones(1, 1))
            Traceback (most recent call last):
                ...
            ValueError: SoftmaxFocalLoss is not suitable for binary tasks. Please use SigmoidFocalLoss instead.

        """
        if predictions.size(1) < 2:
            raise ValueError(
                "SoftmaxFocalLoss is not suitable for binary tasks. "
                "Please use SigmoidFocalLoss instead."
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: {predictions.shape} vs {targets.shape}"
            )
        with amp.autocast(device_type=predictions.device.type, enabled=False):
            predictions = predictions.float()
            targets = targets.float()

            predictions = F.softmax(predictions, dim=1)

            if self._smooth:
                targets = targets.clamp(
                    self._smooth / (predictions.size(1) - 1),
                    1.0 - self._smooth,
                )

            pt = (targets * predictions).sum(dim=1) + self._smooth

            if isinstance(self._alpha, Tensor):
                if self._alpha.size(0) != predictions.size(1):
                    raise ValueError(
                        f"Alpha length {self._alpha.size(0)} does not "
                        f"match number of classes {predictions.size(1)}"
                    )
                alpha_t = self._alpha[targets.argmax(dim=1)]
            else:
                alpha_t = self._alpha

            pt = torch.as_tensor(pt, dtype=torch.float32)
            focal_term = torch.pow(1.0 - pt, self._gamma)
            loss = -alpha_t * focal_term * pt.log()

            if self._reduction == "mean":
                return loss.mean()
            if self._reduction == "sum":
                return loss.sum()
            return loss
