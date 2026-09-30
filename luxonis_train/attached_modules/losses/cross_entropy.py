"""Cross entropy over raw logits."""

from typing import Literal

import torch
from loguru import logger
from torch import Tensor, nn

from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    r"""Cross entropy between logits and class targets.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, C, ...]`` logits
        - ``target`` (``Tensor``): ``[B, C, ...]`` one-hot or ``[B,
          ...]`` class indices

    Outputs:
        - ``Tensor``: scalar, or ``[B, ...]`` when ``reduction`` is
          ``"none"``

    Formula:
        A one-hot target first becomes class indices through ``argmax``
        over the class dimension. For the logits :math:`x` of one
        element, its class index :math:`t`, the number of classes
        :math:`C`, and ``label_smoothing`` :math:`\varepsilon`, the loss
        is

        .. math::

            \ell = -\sum_{c=1}^{C} w_c \, q_c
            \log \frac{e^{x_c}}{\sum_{k=1}^{C} e^{x_k}},
            \qquad
            q_c = (1 - \varepsilon) \, [c = t] + \frac{\varepsilon}{C}

        Here :math:`w_c` is the ``weight`` of class :math:`c`, or ``1``
        without ``weight``. An element with the target ``ignore_index``
        adds nothing. The ``"mean"`` reduction divides the sum of
        :math:`\ell` by the sum of :math:`w_t` over all elements that do
        not have this target.

    References:
        - Source: Wraps `torch.nn.CrossEntropyLoss
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_
          (BSD-3-Clause).
        - License: Apache-2.0 (this project)

    Notes:
        The loss adapts a single channel when ``predictions`` and
        ``target`` have the same number of dimensions. When the class
        dimension of ``predictions`` has size ``1``, the loss puts a
        channel of zero logits in front of it. A target :math:`y` with
        one channel becomes the two channels :math:`1 - y` and
        :math:`y`. Without ``weight`` and ``label_smoothing``, a target
        of ``0`` or ``1`` gives the binary cross entropy of the single
        logit. The loss logs a warning at the first such call.
        `BCEWithLogitsLoss` handles one class directly.

    Example:
        Attached to a ``DDRNetSegmentationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: DDRNetSegmentationHead
              inputs: [DDRNet]
              losses:
                - name: CrossEntropyLoss

    Compatible with:
        - Used by: `ClassificationModel`
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
        weight: list[float] | None = None,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        r"""Initialize the loss and the wrapped ``nn.CrossEntropyLoss``.

        Args:
            weight (list[float] | None): The factor :math:`w_c` of each
                class, one value for each class. ``None`` gives every
                class the factor ``1``.
            ignore_index (int): A class index that adds nothing to the
                loss and to the gradient. A one-hot target becomes
                indices from ``0`` to ``C - 1``. The default ``-100``
                therefore affects only a target of class indices.
            reduction (``Literal["none", "mean", "sum"]``): How to
                reduce the loss of the elements:

                - ``"none"``: return the loss of each element.
                - ``"mean"``: return the weighted mean, as the class
                  formula describes.
                - ``"sum"``: return the sum over all elements.

            label_smoothing (float): The value :math:`\varepsilon`, in
                ``[0, 1]``. The target keeps :math:`1 - \varepsilon` of
                its mass and spreads :math:`\varepsilon` evenly over all
                classes, as in `Rethinking the Inception Architecture
                for Computer Vision <https://arxiv.org/abs/1512.00567>`_.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseLoss`, such as ``final_loss_weight`` and ``node``.

        """
        super().__init__(**kwargs)

        self.criterion = nn.CrossEntropyLoss(
            weight=(torch.tensor(weight) if weight is not None else None),
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self._was_logged = False

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """Compute the cross entropy between logits and class targets.

        When both tensors have the same number of dimensions, the method
        treats ``target`` as one-hot. The class dimension is ``1``, or
        ``0`` for tensors with one dimension. When ``predictions`` has
        one channel, the method first adds a second channel, as the
        class notes describe. It then converts ``target`` to class
        indices with ``argmax`` over the class dimension. For a
        multi-hot target, ``argmax`` selects the first class with the
        highest value.

        Args:
            predictions (``Tensor``): Logits of shape ``[B, C, ...]``,
                the main output of the node.
            target (``Tensor``): One-hot targets of shape
                ``[B, C, ...]``, or class indices of shape ``[B, ...]``.
                The ``classification`` and ``segmentation`` labels have
                the shape ``[B, C, ...]``.

        Returns:
            ``Tensor``: A scalar for the ``"mean"`` and ``"sum"``
            reductions. For ``"none"``, the loss of each element, of
            shape ``[B, ...]``.

        Raises:
            RuntimeError: When ``target`` has neither the number of
                dimensions of ``predictions`` nor one dimension less.

        Example:
            A one-hot target and the same classes as indices give the
            same loss:

            >>> import torch
            >>> loss = CrossEntropyLoss()
            >>> logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
            >>> one_hot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            >>> round(loss(logits, one_hot).item(), 4)
            0.1269
            >>> round(loss(logits, torch.tensor([0, 1])).item(), 4)
            0.1269

        """
        if predictions.ndim == target.ndim:
            ch_dim = 1 if predictions.ndim > 1 else 0
            if predictions.shape[ch_dim] == 1:
                if not self._was_logged:
                    logger.warning(
                        "`CrossEntropyLoss` expects at least 2 classes. "
                        "Attempting to fix by adding a dummy channel. "
                        "If you want to be sure, use `BCEWithLogitsLoss` instead."
                    )
                    self._was_logged = True
                predictions = torch.cat(
                    [torch.zeros_like(predictions), predictions], dim=ch_dim
                )
                if target.shape[ch_dim] == 1:
                    target = torch.cat([1 - target, target], dim=ch_dim)
            target = target.argmax(dim=ch_dim)

        if target.ndim != predictions.ndim - 1:
            raise RuntimeError(
                f"Target tensor dimension should be equal to preds dimension - 1 ({predictions.ndim - 1}) "
                f"but is ({target.ndim})."
            )
        return self.criterion(predictions, target)
