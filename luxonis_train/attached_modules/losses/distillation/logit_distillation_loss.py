"""Distillation of class logits, with a temperature."""

from typing import Literal

import torch
import torch.nn.functional as F
from luxonis_ml.typing import Params
from torch import Size, Tensor
from typing_extensions import override

from luxonis_train.attached_modules.losses.bce_with_logits import (
    BCEWithLogitsLoss,
)
from luxonis_train.attached_modules.losses.sigmoid_focal_loss import (
    SigmoidFocalLoss,
)
from luxonis_train.attached_modules.losses.smooth_bce_with_logits import (
    SmoothBCEWithLogitsLoss,
)
from luxonis_train.typing import Packet

from .base_distillation_loss import BaseDistillationLoss, StudentContext

_SIGMOID_LOSSES = (
    BCEWithLogitsLoss,
    SmoothBCEWithLogitsLoss,
    SigmoidFocalLoss,
)


class LogitDistillationLoss(BaseDistillationLoss):
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
        ``1``, so it would give a loss of ``0``. `derive_params` picks
        ``"sigmoid"`` for a node that trains with a sigmoid loss, such
        as a multi-label head with `BCEWithLogitsLoss`.

        The loss computes in ``float32``, also under mixed precision.
        `setup` raises ``ValueError`` when the student and the teacher
        logits have different shapes.

    Example:
        The automatic recipe adds the loss to classification,
        segmentation and FOMO heads. Set it explicitly on a node:

        .. code-block:: yaml

            - name: ClassificationHead
              losses:
                - name: CrossEntropyLoss
              distillation:
                - name: LogitDistillationLoss
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

    @classmethod
    @override
    def derive_params(cls, student: StudentContext) -> Params:
        """Use the sigmoid when the node trains with a sigmoid loss.

        Args:
            student: The student node of the loss.

        Returns:
            ``activation``, or nothing when no task loss of the node
            applies a sigmoid.

        Example:
            >>> from luxonis_train.attached_modules.losses import (
            ...     BCEWithLogitsLoss,
            ... )
            >>> context = StudentContext(
            ...     losses={"bce": BCEWithLogitsLoss()}, levels_read=None
            ... )
            >>> LogitDistillationLoss.derive_params(context)
            {'activation': 'sigmoid'}

        """
        if any(
            isinstance(loss, _SIGMOID_LOSSES)
            for loss in student.losses.values()
        ):
            return {"activation": "sigmoid"}
        return {}

    @override
    def setup(
        self, student_shapes: Packet[Size], teacher_shapes: Packet[Size]
    ) -> None:
        """Check that the student and the teacher logits match.

        Args:
            student_shapes: The output shapes of the node.
            teacher_shapes: The output shapes of the teacher node.

        Raises:
            ValueError: When the teacher node gives no logits, or logits
                of another shape.

        """
        super().setup(student_shapes, teacher_shapes)
        key = self.task.main_output
        if student_shapes[key] != teacher_shapes[key]:
            raise ValueError(
                f"'{self.name}' compares the student logits "
                f"{student_shapes[key]} with the teacher logits "
                f"{teacher_shapes[key]}, but the shapes differ."
            )

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
            >>> LogitDistillationLoss(temperature=4.0)(logits, logits).item()
            0.0

            A higher temperature puts more weight on the classes that
            the teacher ranks low:

            >>> teacher = torch.tensor([[3.0, 0.0, -1.0]])
            >>> round(LogitDistillationLoss()(logits, teacher).item(), 4)
            0.2142
            >>> round(
            ...     LogitDistillationLoss(temperature=4.0)(
            ...         logits, teacher
            ...     ).item(),
            ...     4,
            ... )
            0.4983

            One class uses the sigmoid, so the loss is not zero:

            >>> student, teacher = torch.tensor([[0.0]]), torch.tensor([[2.0]])
            >>> round(LogitDistillationLoss()(student, teacher).item(), 4)
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
