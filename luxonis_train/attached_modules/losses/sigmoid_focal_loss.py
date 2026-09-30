"""Binary focal loss, which lowers the weight of the easy examples."""

from typing import Literal

from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks


class SigmoidFocalLoss(BaseLoss):
    r"""Focal loss on independent sigmoid outputs.

    The loss treats each element of the logits as one binary decision,
    so it fits binary and multi-label tasks. The focal factor lowers the
    loss of the elements that the model already predicts well.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, C, ...]`` logits
        - ``target`` (``Tensor``): same shape, float values in
          ``[0, 1]``

    Outputs:
        - ``Tensor``: scalar, or ``[B, C, ...]`` when ``reduction`` is
          ``"none"``

    Formula:
        For a logit :math:`x` with the target :math:`y`, let
        :math:`p = \sigma(x)` and :math:`p_t = p y + (1 - p)(1 - y)`.
        The loss of one element is

        .. math::

            \ell = -\alpha_t \, (1 - p_t)^{\gamma}
            \left[ y \log p + (1 - y) \log (1 - p) \right]

        Here :math:`\alpha_t = \alpha y + (1 - \alpha)(1 - y)`, and an
        ``alpha`` of ``-1`` gives :math:`\alpha_t = 1`. ``reduction``
        then takes the mean or the sum over all elements, or keeps the
        loss of each element.

    References:
        - Source: Wraps `torchvision.ops.sigmoid_focal_loss
          <https://docs.pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html>`_
          (BSD-3-Clause).
        - Paper: `Focal Loss for Dense Object Detection
          <https://arxiv.org/abs/1708.02002>`_
        - License: Apache-2.0 (this project)

    Notes:
        The class wraps ``torchvision.ops.sigmoid_focal_loss`` and adds
        no checks of its own.

    Example:
        Attached to a ``DDRNetSegmentationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: DDRNetSegmentationHead
              inputs: [DDRNet]
              losses:
                - name: SigmoidFocalLoss

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
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        r"""Initialize the loss and store the focal parameters.

        The constructor does not check the values. `forward` passes them
        to ``torchvision``, which raises ``ValueError`` for a value that
        it does not accept.

        Args:
            alpha (float): The weight :math:`\alpha` of the positive
                elements, in ``[0, 1]``. The negative elements get
                :math:`1 - \alpha`. ``-1`` turns the weighting off.
            gamma (float): The exponent of the focal factor
                :math:`(1 - p_t)`. A larger value lowers the loss of the
                well-predicted elements more. ``0`` gives the binary
                cross entropy, weighted by ``alpha``.
            reduction (``Literal["none", "mean", "sum"]``): How to
                reduce the loss of the elements:

                - ``"none"``: return the loss of each element.
                - ``"mean"``: return the mean over all elements.
                - ``"sum"``: return the sum over all elements.

            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseLoss`, such as ``final_loss_weight`` and ``node``.

        """
        super().__init__(**kwargs)

        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """Compute the sigmoid focal loss between logits and targets.

        ``torchvision`` raises ``ValueError`` when the shapes differ,
        and when ``alpha`` or ``reduction`` has a value that it does not
        accept.

        Args:
            predictions (``Tensor``): Logits of shape ``[B, C, ...]``,
                the main output of the node.
            target (``Tensor``): Float targets in ``[0, 1]``, of the
                same shape as ``predictions``.

        Returns:
            ``Tensor``: A scalar for the ``"mean"`` and ``"sum"``
            reductions. For ``"none"``, the loss of each element, of
            shape ``[B, C, ...]``.

        Example:
            With ``alpha=-1`` and ``gamma=0``, the loss is the binary
            cross entropy. The default focal factor makes it smaller:

            >>> import torch
            >>> import torch.nn.functional as F
            >>> logits = torch.tensor([[2.0, -1.0]])
            >>> target = torch.tensor([[1.0, 0.0]])
            >>> bce = F.binary_cross_entropy_with_logits(logits, target)
            >>> plain = SigmoidFocalLoss(alpha=-1.0, gamma=0.0)
            >>> torch.allclose(plain(logits, target), bce)
            True
            >>> bool(SigmoidFocalLoss()(logits, target) < bce)
            True

        """
        return sigmoid_focal_loss(
            predictions,
            target,
            alpha=self._alpha,
            gamma=self._gamma,
            reduction=self._reduction,
        )
