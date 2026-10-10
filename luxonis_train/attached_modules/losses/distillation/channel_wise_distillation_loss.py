"""Channel-wise distillation of feature maps."""

import torch
import torch.nn.functional as F
from luxonis_ml.typing import Params
from torch import Size, Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.typing import Packet

from .base_distillation_loss import BaseDistillationLoss, StudentContext


class ChannelWiseDistillationLoss(BaseDistillationLoss):
    r"""Match the spatial distribution of every teacher channel.

    The loss turns each channel of a feature map into a probability
    distribution over its pixels, with a softmax at temperature
    :math:`\tau`. It then measures, channel by channel, how far the
    student distribution is from the teacher distribution. The loss
    highlights where each channel responds, not how strongly, so it
    works between models of different widths.

    Inputs:
        - ``features`` (``Tensor | list[Tensor]``): ``[B, C_s, H, W]``
          feature maps of the node, one tensor for each level
        - ``teacher_features`` (``Tensor | list[Tensor]``):
          ``[B, C_t, H_t, W_t]`` feature maps of the matched teacher
          node, with the same number of levels

    Outputs:
        - ``Tensor``: scalar, the sum over the levels

    Formula:
        For one level, a 1x1 adapter maps the student map :math:`s` to
        :math:`C_t` channels. For each channel :math:`c`:

        .. math::

            \ell = \frac{\tau^2}{B \, C_t} \sum_{b, c} \mathrm{KL}\left(
            \mathrm{softmax}(t_{b,c} / \tau) \,\|\,
            \mathrm{softmax}(s_{b,c} / \tau) \right)

        The softmax runs over the :math:`H \times W` pixels of the
        channel.

    References:
        - Source: C. Shu, Y. Liu, J. Gao, Z. Yan, C. Shen,
          `Channel-wise Knowledge Distillation for Dense Prediction
          <https://arxiv.org/abs/2011.13256>`_, ICCV 2021.
        - License: Apache-2.0 (this project)

    Notes:
        Without ``levels``, the loss compares the levels that the nodes
        fed by its node read, as `derive_params` sets them.

        `setup` creates one adapter for each level: a 1x1 convolution
        with batch norm when the channel counts differ, and an identity
        otherwise. When the spatial sizes differ, the loss resizes the
        teacher map to the student map with bilinear interpolation.

        The loss computes in ``float32``, also under mixed precision.

    Example:
        The automatic recipe adds the loss to the node that feeds a
        matched head. Set it explicitly on a neck:

        .. code-block:: yaml

            - name: RepPANNeck
              distillation:
                - name: ChannelWiseDistillationLoss
                  params:
                    tau: 1.0

    Compatible with:
        - Nodes: any node whose ``features`` output holds feature maps,
          such as backbones and necks

    """

    def __init__(
        self, tau: float = 1.0, levels: list[int] | None = None, **kwargs
    ):
        """Initialize the loss.

        Args:
            tau: The temperature of the softmax.
            levels: The indices of the levels to compare, in both
                nodes. ``None`` compares all of them.
            **kwargs: Keyword arguments forwarded to
                `BaseDistillationLoss`.

        """
        super().__init__(**kwargs)
        self.tau = tau
        self.levels = levels
        self.adapters: nn.ModuleList | None = None

    @classmethod
    @override
    def derive_params(cls, student: StudentContext) -> Params:
        """Compare the levels that the nodes fed by the node read.

        Args:
            student: The student node of the loss.

        Returns:
            ``levels``, or nothing when no node reads the feature maps.

        Example:
            >>> context = StudentContext(losses={}, levels_read=[2, 3])
            >>> ChannelWiseDistillationLoss.derive_params(context)
            {'levels': [2, 3]}

        """
        if student.levels_read is None:
            return {}
        return {"levels": student.levels_read}

    @override
    def setup(
        self, student_shapes: Packet[Size], teacher_shapes: Packet[Size]
    ) -> nn.Module:
        """Create one channel adapter for each compared level.

        Args:
            student_shapes: The output shapes of the node.
            teacher_shapes: The output shapes of the teacher node.

        Returns:
            The adapters, in level order.

        Raises:
            ValueError: When the teacher node gives no ``features``,
                when it has fewer levels than ``levels`` names, or when
                the two nodes give a different number of levels.

        """
        super().setup(student_shapes, teacher_shapes)
        student = self._select(student_shapes["features"])
        try:
            teacher = self._select(teacher_shapes["features"])
        except IndexError as error:
            raise ValueError(
                f"'{self.name}' compares the levels {self.levels}, but the "
                "teacher node has fewer levels."
            ) from error
        if len(student) != len(teacher):
            raise ValueError(
                f"'{self.name}' compares {len(student)} student levels "
                f"with {len(teacher)} teacher levels. Set `levels` so "
                "that both nodes give the same number."
            )
        self.adapters = nn.ModuleList(
            channel_adapter(s[1], t[1])
            for s, t in zip(student, teacher, strict=True)
        )
        return self.adapters

    def forward(
        self,
        features: Tensor | list[Tensor],
        teacher_features: Tensor | list[Tensor],
    ) -> Tensor:
        """Compute the distillation loss of one batch.

        Args:
            features: The student maps.
            teacher_features: The teacher maps.

        Returns:
            The scalar loss.

        Raises:
            RuntimeError: When a level has different channel counts and
                `setup` has not run.

        Examples:
            >>> import torch
            >>> student = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
            >>> teacher = torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]])
            >>> ChannelWiseDistillationLoss()(student, student).item()
            0.0
            >>> round(
            ...     ChannelWiseDistillationLoss()(student, teacher).item(), 4
            ... )
            0.1141

        """
        student = self._select(features)
        teacher = self._select(teacher_features)
        with torch.autocast(student[0].device.type, enabled=False):
            losses = [
                self._level_loss(self._adapt(i, s.float()), t.float())
                for i, (s, t) in enumerate(zip(student, teacher, strict=True))
            ]
        return torch.stack(losses).sum()

    def _select(self, values: Tensor | Size | list) -> list:
        values = values if isinstance(values, list) else [values]
        if self.levels is None:
            return values
        return [values[i] for i in self.levels]

    def _adapt(self, index: int, student: Tensor) -> Tensor:
        if self.adapters is not None:
            return self.adapters[index](student)
        return student

    def _level_loss(self, student: Tensor, teacher: Tensor) -> Tensor:
        """Compute the loss of one level, after the adapter."""
        if student.shape[1] != teacher.shape[1]:
            raise RuntimeError(
                f"'{self.name}' got {student.shape[1]} student channels "
                f"and {teacher.shape[1]} teacher channels, but has no "
                "adapter. Call `setup` first."
            )
        if student.shape[-2:] != teacher.shape[-2:]:
            teacher = F.interpolate(
                teacher,
                size=student.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        n, c = student.shape[:2]
        log_student = F.log_softmax(student.reshape(n * c, -1) / self.tau, 1)
        log_teacher = F.log_softmax(teacher.reshape(n * c, -1) / self.tau, 1)
        divergence = F.kl_div(
            log_student, log_teacher, log_target=True, reduction="sum"
        )
        return divergence * self.tau**2 / (n * c)


def channel_adapter(in_channels: int, out_channels: int) -> nn.Module:
    """Map ``in_channels`` feature maps to ``out_channels`` channels.

    Args:
        in_channels: The channels of the student map.
        out_channels: The channels of the teacher map.

    Returns:
        A 1x1 convolution with batch norm and no activation, or
        ``nn.Identity`` when the counts are equal.

    Example:
        >>> channel_adapter(64, 64)
        Identity()

    """
    if in_channels == out_channels:
        return nn.Identity()
    return ConvBlock(
        in_channels, out_channels, kernel_size=1, activation=False
    )
