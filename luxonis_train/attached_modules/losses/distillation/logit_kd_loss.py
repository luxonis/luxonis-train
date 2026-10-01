"""Distillation of class logits, with a temperature."""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .base_distillation_loss import BaseDistillationLoss


class LogitKDLoss(BaseDistillationLoss):
    r"""Match the softened class distribution of a teacher head.

    The loss divides the student and the teacher logits by the
    temperature :math:`T`. It then measures how far the student
    distribution is from the teacher distribution, and scales the result
    by :math:`T^2`, so the gradient keeps its size when :math:`T`
    changes.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, C, ...]`` logits of the
          head, its main output
        - ``teacher`` (``Tensor``): the logits of the matched teacher
          head, same shape

    Outputs:
        - ``Tensor``: scalar

    Formula:
        With ``activation`` ``"softmax"``, the classes along ``dim``
        form one distribution:

        .. math::

            \ell = T^2 \, \mathrm{KL}\left(
            \mathrm{softmax}(t / T) \,\|\, \mathrm{softmax}(s / T)
            \right)

        With ``"sigmoid"``, each class is its own Bernoulli
        distribution, and the loss sums their Kullback-Leibler
        divergences over ``dim``:

        .. math::

            \ell = T^2 \sum_c \mathrm{KL}\left(
            \sigma(t_c / T) \,\|\, \sigma(s_c / T) \right)

        The loss takes the mean over all the other dimensions, such as
        the batch and the pixels.

    References:
        - Source: G. Hinton, O. Vinyals, J. Dean, `Distilling the
          Knowledge in a Neural Network
          <https://arxiv.org/abs/1503.02531>`_, 2015.
        - License: Apache-2.0 (this project)

    Notes:
        ``"auto"`` picks ``"sigmoid"`` when ``dim`` has one class and
        ``"softmax"`` otherwise. A softmax over one class is always
        ``1``, so it would give a loss of ``0``. Set ``"sigmoid"`` for a
        multi-label head.

        The loss computes in ``float32``, also under mixed precision.
        It raises ``RuntimeError`` when the two tensors have different
        shapes.

    Example:
        The automatic recipe adds the loss to classification,
        segmentation and FOMO heads. Set it explicitly on a node:

        .. code-block:: yaml

            - name: ClassificationHead
              losses:
                - name: CrossEntropyLoss
              distillation:
                - name: LogitKDLoss
                  params:
                    temperature: 4.0

    Compatible with:
        - Nodes:

          - `ClassificationHead`
          - `TransformerClassificationHead`
          - `SegmentationHead`
          - `BiSeNetHead`
          - `DDRNetSegmentationHead`
          - `TransformerSegmentationHead`
          - `FOMOHead`

    """

    def __init__(
        self,
        temperature: float = 1.0,
        activation: Literal["auto", "softmax", "sigmoid"] = "auto",
        dim: int = 1,
        **kwargs,
    ):
        """Initialize the loss.

        Args:
            temperature: The temperature :math:`T`. A larger value gives
                softer distributions.
            activation: How the logits become probabilities, as the
                class docstring describes.
            dim: The dimension of the classes.
            **kwargs: Keyword arguments forwarded to
                `BaseDistillationLoss`.

        """
        super().__init__(**kwargs)
        self.temperature = temperature
        self.activation = activation
        self.dim = dim

    def forward(self, predictions: Tensor, teacher: Tensor) -> Tensor:
        """Compute the distillation loss of one batch.

        Args:
            predictions: The student logits.
            teacher: The teacher logits, of the same shape.

        Returns:
            The scalar loss.

        Raises:
            RuntimeError: When the shapes differ.

        Examples:
            The loss is zero when the student matches the teacher:

            >>> import torch
            >>> logits = torch.tensor([[1.0, 0.0, -1.0]])
            >>> LogitKDLoss(temperature=4.0)(logits, logits).item()
            0.0

            A higher temperature puts more weight on the classes that
            the teacher ranks low:

            >>> teacher = torch.tensor([[3.0, 0.0, -1.0]])
            >>> round(LogitKDLoss()(logits, teacher).item(), 4)
            0.2142
            >>> round(LogitKDLoss(temperature=4.0)(logits, teacher).item(), 4)
            0.4983

            One class uses the sigmoid, so the loss is not zero:

            >>> student, teacher = torch.tensor([[0.0]]), torch.tensor([[2.0]])
            >>> round(LogitKDLoss()(student, teacher).item(), 4)
            0.3278

        """
        if predictions.shape != teacher.shape:
            raise RuntimeError(
                f"Student logits {tuple(predictions.shape)} and teacher "
                f"logits {tuple(teacher.shape)} have different shapes."
            )
        with torch.autocast(predictions.device.type, enabled=False):
            student = predictions.float() / self.temperature
            target = teacher.float() / self.temperature
            if self._uses_sigmoid(student):
                divergence = _bernoulli_kl(student, target)
            else:
                divergence = F.kl_div(
                    F.log_softmax(student, dim=self.dim),
                    F.log_softmax(target, dim=self.dim),
                    log_target=True,
                    reduction="none",
                )
            loss = divergence.sum(dim=self.dim).mean()
        return loss * self.temperature**2

    def _uses_sigmoid(self, logits: Tensor) -> bool:
        if self.activation == "auto":
            return logits.shape[self.dim] == 1
        return self.activation == "sigmoid"


def _bernoulli_kl(student: Tensor, teacher: Tensor) -> Tensor:
    # KL(p || q) = H(p, q) - H(p); both cross entropies take logits.
    target = torch.sigmoid(teacher)
    cross_entropy = F.binary_cross_entropy_with_logits(
        student, target, reduction="none"
    )
    entropy = F.binary_cross_entropy_with_logits(
        teacher, target, reduction="none"
    )
    return cross_entropy - entropy
