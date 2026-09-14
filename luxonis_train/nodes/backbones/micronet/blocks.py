"""The blocks of the MicroNet backbone."""

from typing import Literal

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvBlock


class MicroBlock(nn.Module):
    r"""The basic block of MicroNet.

    The block expands the input to
    ``in_channels * expand_ratio[0] * expand_ratio[1]`` channels.
    ``groups_1`` and ``groups_2`` select one of three layouts:

    - A *lite* block, when ``groups_1[0]`` is ``0``. A
      `DepthSpatialSepConv` expands the channels and applies the stride.
      A grouped :math:`1 \times 1` convolution projects them to
      ``out_channels``.
    - A *transition* block, when ``groups_2[1]`` is ``0`` and
      ``groups_1[0]`` is not ``0``. A grouped :math:`1 \times 1`
      convolution expands the channels. The block has no depthwise
      convolution and no projection. Its output keeps the expanded
      channels.
    - A *full* block, in all other cases. A grouped :math:`1 \times 1`
      convolution expands the channels. A `DepthSpatialSepConv` without
      expansion applies the stride. A second grouped :math:`1 \times 1`
      convolution projects the channels to ``out_channels``.

    Each convolution has a batch norm and no bias. A `DYShiftMax`, a
    ``ReLU6``, or no activation follows each :math:`1 \times 1`
    convolution and each `DepthSpatialSepConv`, as ``dy_shift`` selects.
    `ChannelShuffle` layers mix the channels of the groups. The block
    adds its input to its output when ``stride`` is ``1`` and
    ``out_channels`` is equal to ``in_channels``.

    Example:
        >>> import torch
        >>> x = torch.zeros(2, 8, 16, 16)
        >>> lite = MicroBlock(
        ...     8, 16, stride=2, groups_1=(0, 8), groups_2=(4, 4)
        ... )
        >>> lite(x).shape
        torch.Size([2, 16, 8, 8])

        A transition block ignores ``out_channels`` and ``stride`` in its
        layers:

        >>> transition = MicroBlock(
        ...     8,
        ...     16,
        ...     stride=2,
        ...     expand_ratio=(1, 6),
        ...     groups_1=(4, 4),
        ...     groups_2=(0, 0),
        ... )
        >>> transition(x).shape
        torch.Size([2, 48, 16, 16])

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: tuple[int, int] = (2, 2),
        groups_1: tuple[int, int] = (0, 6),
        groups_2: tuple[int, int] = (1, 1),
        dy_shift: tuple[int, int, int] = (2, 0, 1),
        reduction_factor: int = 1,
        init_a: tuple[float, float] = (1.0, 1.0),
        init_b: tuple[float, float] = (0.0, 0.0),
    ):
        r"""Build the layers of the lite, transition, or full layout.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels of a lite
                or a full block. The layers of a transition block do not
                read it. The residual check reads it in all layouts.
            kernel_size (int): The kernel size :math:`k` of the
                `DepthSpatialSepConv`, which is a :math:`k \times 1` and
                a :math:`1 \times k` convolution. A transition block
                ignores it.
            stride (int): The stride of the `DepthSpatialSepConv`. The
                layers of a transition block ignore it.
            expand_ratio (tuple[int, int]): The two channel multipliers
                of the expansion. A lite block applies one multiplier in
                each half of its `DepthSpatialSepConv`. The other layouts
                apply the product in the expansion :math:`1 \times 1`
                convolution.
            groups_1 (tuple[int, int]): The first value is the number of
                groups of the expansion :math:`1 \times 1` convolution.
                ``0`` selects a lite block. The second value sets the
                groups of the `DYShiftMax` layers before the projection
                and of the `ChannelShuffle` after the first activation.
                In the `DYShiftMax` after the depthwise convolution of a
                full block, the groups are the expanded channels divided
                by the value, when the value is not ``1``. In a transition
                block, the value sets the groups of its only `DYShiftMax`.
            groups_2 (tuple[int, int]): The first value is the number of
                groups of the projection :math:`1 \times 1` convolution.
                The second value sets the groups of the last `DYShiftMax`
                and of the `ChannelShuffle` after it. ``0`` selects a
                transition block when ``groups_1[0]`` is not ``0``.
            dy_shift (tuple[int, int, int]): The activations after the
                expansion convolution, after the depthwise convolution,
                and after the projection. In the first two positions, a
                positive value selects `DYShiftMax`, and other values
                select ``ReLU6``. ``2`` selects two branches, and another
                positive value selects one branch. In the last position,
                a positive value selects a one-branch `DYShiftMax`, and
                other values select no activation. A lite block ignores
                the first value. A transition block reads only the last
                value, for the activation after its expansion. In a lite
                or a full block, values other than ``0`` also add a
                `ChannelShuffle` after an activation. In a lite block,
                the second value adds one with ``C // 2`` groups, where
                ``C`` is the number of expanded channels. In a full block,
                the first two values share one after the depthwise
                activation. It has ``C // 4`` groups when both values are
                not ``0``, and ``C // 2`` groups when only one is not
                ``0``. The last value adds one with ``out_channels // 2``
                groups. In a lite block, it does so only when
                ``out_channels`` is even.
            reduction_factor (int): The reduction of the squeeze network
                of `DYShiftMax`. The activations use a ``reduction`` of
                ``8 * reduction_factor``. The last activation of a lite
                block uses ``4 * reduction_factor``. The last activation
                of a full block also does, when ``out_channels`` is
                smaller than the expanded channels.
            init_a (tuple[float, float]): The offsets for the weights of
                the input in `DYShiftMax`. The activations after the
                expansion and after the depthwise convolution use them.
                The last activation, and the activation of a transition
                block, use ``(1.0, 0.0)`` instead.
            init_b (tuple[float, float]): The offsets for the weights of
                the shifted input in `DYShiftMax`. The same activations
                as for ``init_a`` use them. The others use ``(0.0, 0.0)``.

        """
        super().__init__()

        self._use_residual = stride == 1 and in_channels == out_channels
        self._expand_ratio = expand_ratio
        use_dy1, use_dy2, use_dy3 = dy_shift
        group1, group2 = groups_2
        reduction = 8 * reduction_factor
        intermediate_channels = in_channels * expand_ratio[0] * expand_ratio[1]

        if groups_1[0] == 0:
            self.layers = self._create_lite_block(
                in_channels,
                out_channels,
                intermediate_channels,
                kernel_size,
                stride,
                groups_1[1],
                group1,
                group2,
                use_dy2,
                use_dy3,
                reduction,
                init_a,
                init_b,
            )
        elif group2 == 0:
            self.layers = self._create_transition_block(
                in_channels,
                intermediate_channels,
                groups_1[0],
                groups_1[1],
                use_dy3,
                reduction,
            )
        else:
            self.layers = self._create_full_block(
                in_channels,
                out_channels,
                intermediate_channels,
                kernel_size,
                stride,
                groups_1,
                group1,
                group2,
                use_dy1,
                use_dy2,
                use_dy3,
                reduction,
                init_a,
                init_b,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the layers, and add the input for a residual connection.

        Args:
            inputs (``Tensor``): The input of shape
                ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: The output of shape ``[B, C, H', W']``. ``C`` is
            ``out_channels`` for a lite or a full block, and the expanded
            number of channels for a transition block. For an odd
            ``kernel_size``, ``H'`` is ``ceil(H / stride)`` and ``W'`` is
            ``ceil(W / stride)``. A transition block keeps ``H`` and
            ``W``. With the residual connection, the output is the sum
            of the layer output and ``inputs``. For a transition block,
            this sum raises ``RuntimeError`` unless the expanded number
            of channels is equal to ``in_channels``.

        """
        out = self.layers(inputs)
        if self._use_residual:
            out += inputs
        return out

    def _create_lite_block(
        self,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int,
        stride: int,
        group1: int,
        group2: int,
        group3: int,
        use_dy2: int,
        use_dy3: int,
        reduction: int,
        init_a: tuple[float, float],
        init_b: tuple[float, float],
    ) -> nn.Sequential:
        return nn.Sequential(
            DepthSpatialSepConv(
                in_channels, self._expand_ratio, kernel_size, stride
            ),
            DYShiftMax(
                intermediate_channels,
                intermediate_channels,
                init_a,
                init_b,
                use_dy2 == 2,
                group1,
                reduction,
            )
            if use_dy2 > 0
            else nn.ReLU6(True),
            ChannelShuffle(group1),
            ChannelShuffle(intermediate_channels // 2)
            if use_dy2 != 0
            else nn.Sequential(),
            ConvBlock(
                in_channels=intermediate_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=group2,
                activation=None,
            ),
            DYShiftMax(
                out_channels,
                out_channels,
                (1.0, 0.0),
                (0.0, 0.0),
                False,
                group3,
                reduction // 2,
            )
            if use_dy3 > 0
            else nn.Sequential(),
            ChannelShuffle(group3),
            ChannelShuffle(out_channels // 2)
            if out_channels % 2 == 0 and use_dy3 != 0
            else nn.Sequential(),
        )

    def _create_transition_block(
        self,
        in_channels: int,
        intermediate_channels: int,
        group1: int,
        group2: int,
        use_dy3: int,
        reduction: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=1,
                groups=group1,
                activation=None,
            ),
            DYShiftMax(
                intermediate_channels,
                intermediate_channels,
                (1.0, 0.0),
                (0.0, 0.0),
                False,
                group2,
                reduction,
            )
            if use_dy3 > 0
            else nn.Sequential(),
        )

    def _create_full_block(
        self,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int,
        stride: int,
        groups_1: tuple[int, int],
        group1: int,
        group2: int,
        use_dy1: int,
        use_dy2: int,
        use_dy3: int,
        reduction: int,
        init_a: tuple[float, float],
        init_b: tuple[float, float],
    ) -> nn.Sequential:
        return nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=1,
                groups=groups_1[0],
                activation=None,
            ),
            DYShiftMax(
                intermediate_channels,
                intermediate_channels,
                init_a,
                init_b,
                use_dy1 == 2,
                groups_1[1],
                reduction,
            )
            if use_dy1 > 0
            else nn.ReLU6(True),
            ChannelShuffle(groups_1[1]),
            DepthSpatialSepConv(
                intermediate_channels, (1, 1), kernel_size, stride
            ),
            DYShiftMax(
                intermediate_channels,
                intermediate_channels,
                init_a,
                init_b,
                use_dy2 == 2,
                groups_1[1],
                reduction,
                True,
            )
            if use_dy2 > 0
            else nn.ReLU6(True),
            ChannelShuffle(intermediate_channels // 4)
            if use_dy1 != 0 and use_dy2 != 0
            else nn.Sequential()
            if use_dy1 == 0 and use_dy2 == 0
            else ChannelShuffle(intermediate_channels // 2),
            ConvBlock(
                in_channels=intermediate_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=group1,
                activation=None,
            ),
            DYShiftMax(
                out_channels,
                out_channels,
                (1.0, 0.0),
                (0.0, 0.0),
                False,
                group2,
                reduction=reduction // 2
                if out_channels < intermediate_channels
                else reduction,
            )
            if use_dy3 > 0
            else nn.Sequential(),
            ChannelShuffle(group2),
            ChannelShuffle(out_channels // 2)
            if use_dy3 != 0
            else nn.Sequential(),
        )


class ChannelShuffle(nn.Module):
    """Channel shuffle that interleaves the channels of the groups.

    The module splits the channels into ``groups`` groups of equal size.
    The output takes the first channel of each group, then the second
    channel of each group, and so on. A grouped convolution after the
    shuffle then reads channels from all groups. With ``groups`` equal
    to ``1`` or to the number of channels, the order does not change.

    Example:
        >>> import torch
        >>> x = torch.arange(6.0).view(1, 6, 1, 1)
        >>> ChannelShuffle(3)(x).flatten().tolist()
        [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]

    """

    def __init__(self, groups: int):
        """Store the number of groups.

        Args:
            groups (int): The number of groups. The number of input
                channels must be a multiple of it.

        """
        super().__init__()
        self._groups = groups

    def forward(self, x: Tensor) -> Tensor:
        """Interleave the channels of the groups.

        Args:
            x (``Tensor``): The input of shape ``[B, C, H, W]``. ``C``
                must be a multiple of ``groups``. Otherwise, the reshape
                raises ``RuntimeError``.

        Returns:
            ``Tensor``: The input with its channels in the new order, of
            the same shape.

        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self._groups
        x = x.view(batch_size, self._groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batch_size, -1, height, width)


class DYShiftMax(nn.Module):
    r"""Dynamic Shift-Max activation of MicroNet.

    The activation mixes each channel with one channel of the next
    group. The shifted input :math:`\tilde{x}` takes channel ``c + 1``
    of group ``g + 1`` for channel ``c`` of group ``g``. Both indices
    wrap around. With two branches, the output is

    .. math::

        y = \max\left(a_1 x + b_1 \tilde{x},\; a_2 x + b_2 \tilde{x}\right)

    With one branch, the output is :math:`y = a_1 x + b_1 \tilde{x}`.

    A squeeze network computes the coefficients from the input. It
    averages each channel over the height and the width. Two linear
    layers with a ``ReLU`` between them and a hard sigmoid follow. The
    module maps the result from ``[0, 1]`` to ``[-2, 2]`` and adds the
    offsets ``init_a`` and ``init_b``. Each coefficient has
    ``out_channels`` values for each sample.

    Example:
        ``index`` holds the input channel that each channel of
        :math:`\tilde{x}` takes. With 8 channels in 2 groups, channel
        ``0`` takes channel ``5``, the second channel of the second
        group.

        >>> import torch
        >>> act = DYShiftMax(8, 8, groups=2)
        >>> act.index.tolist()
        [5, 6, 7, 4, 1, 2, 3, 0]
        >>> act(torch.ones(2, 8, 4, 4)).shape
        torch.Size([2, 8, 4, 4])

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_a: tuple[float, float] = (0.0, 0.0),
        init_b: tuple[float, float] = (0.0, 0.0),
        use_relu: bool = True,
        groups: int = 6,
        reduction: int = 4,
        expansion: bool = False,
    ):
        r"""Initialize the squeeze network and the channel shift.

        Args:
            in_channels (int): The number of input channels. It must be
                a multiple of the number of groups.
            out_channels (int): The number of channels of each
                coefficient. `forward` needs it equal to ``in_channels``
                or to ``1``. With ``1``, all channels of a sample share
                each coefficient.
            init_a (tuple[float, float]): The offsets for :math:`a_1`
                and :math:`a_2`. `forward` adds ``init_a[0]`` to
                :math:`a_1`. It does not read ``init_a[1]``, because it
                adds ``init_b[1]`` to :math:`a_2`.
            init_b (tuple[float, float]): The offsets for :math:`b_1`
                and :math:`b_2`. ``init_b[1]`` also goes to :math:`a_2`.
            use_relu (bool): ``True`` selects two branches and their
                maximum, a dynamic form of ``ReLU``. ``False`` selects
                one branch without a maximum.
            groups (int): The number of channel groups for the shift.
                With ``1``, the shift moves the channels by one position.
            reduction (int): The divisor of ``in_channels`` for the
                hidden layer of the squeeze network. The module rounds
                ``in_channels // reduction`` to the nearest multiple of
                ``4``, with a minimum of ``4``. When the rounding goes
                more than 10% down, it adds ``4``.
            expansion (bool): When ``True`` and ``groups`` is not ``1``,
                the number of groups is ``in_channels // groups``. Then
                ``groups`` is the number of channels in each group.

        """
        super().__init__()
        self._exp: Literal[2, 4] = 4 if use_relu else 2
        self._init_a = init_a
        self._init_b = init_b
        self._out_channels = out_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        squeeze_channels = _make_divisible(in_channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeeze_channels),
            nn.ReLU(),
            nn.Linear(squeeze_channels, out_channels * self._exp),
            nn.Hardsigmoid(),
        )

        if groups != 1 and expansion:
            groups = in_channels // groups

        channels_per_group = in_channels // groups
        index = torch.arange(in_channels).view(1, in_channels, 1, 1)
        index = index.view(1, groups, channels_per_group, 1, 1)
        index_groups = torch.split(index, [1, groups - 1], dim=1)
        index_groups = torch.cat([index_groups[1], index_groups[0]], dim=1)
        index_splits = torch.split(
            index_groups, [1, channels_per_group - 1], dim=2
        )
        index_splits = torch.cat([index_splits[1], index_splits[0]], dim=2)
        self._index = index_splits.view(in_channels).long()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the activation with the coefficients of the input.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: The activated input, of the same shape.

        Raises:
            RuntimeError: When ``exp`` is not ``2`` or ``4``. The
                constructor sets only these two values.

        """
        batch_size, channels, _, _ = x.shape
        x_out = x

        y = self.avg_pool(x).view(batch_size, channels)
        y: Tensor = self.fc(y).view(batch_size, -1, 1, 1)
        y = (y - 0.5) * 4.0

        x2 = x_out[:, self._index, :, :]

        if self._exp == 4:
            a1, b1, a2, b2 = torch.split(y, self._out_channels, dim=1)

            a1 = a1 + self._init_a[0]
            a2 = a2 + self._init_b[1]
            b1 = b1 + self._init_b[0]
            b2 = b2 + self._init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)

        elif self._exp == 2:
            a1, b1 = y.split(self._out_channels, dim=1)
            a1 = a1 + self._init_a[0]
            b1 = b1 + self._init_b[0]
            out = x_out * a1 + x2 * b1
        else:
            raise RuntimeError("Expansion should be 2 or 4.")

        return out


def _make_divisible(value: int, divisor: int) -> int:
    min_value = divisor
    new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * value:
        new_v += divisor
    return new_v


class SpatialSepConvSF(nn.Module):
    r"""Spatially separable convolution with a channel shuffle.

    A :math:`k \times 1` convolution maps the input to ``outs[0]``
    channels and applies the stride along the height. A
    :math:`1 \times k` convolution with ``outs[0]`` groups multiplies
    the channels by ``outs[1]`` and applies the stride along the width.
    A batch norm follows each convolution. A `ChannelShuffle` with
    ``outs[0]`` groups ends the block. The convolutions have no bias,
    and the block has no activation.

    Example:
        >>> import torch
        >>> conv = SpatialSepConvSF(3, (4, 2), kernel_size=3, stride=2)
        >>> conv(torch.zeros(1, 3, 32, 32)).shape
        torch.Size([1, 8, 16, 16])

    """

    def __init__(
        self,
        in_channels: int,
        outs: tuple[int, int],
        kernel_size: int,
        stride: int,
    ):
        r"""Initialize the two convolutions and the channel shuffle.

        Args:
            in_channels (int): The number of input channels.
            outs (tuple[int, int]): The channel layout. The first value
                is the number of output channels of the
                :math:`k \times 1` convolution. It is also the number of
                groups of the :math:`1 \times k` convolution and of the
                shuffle. The second value is the channel multiplier of
                the :math:`1 \times k` convolution.
            kernel_size (int): The kernel size :math:`k`. The padding is
                ``kernel_size // 2``.
            stride (int): The stride along the height and the width.

        """
        super().__init__()
        out_channels1, out_channels2 = outs
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels1,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(kernel_size // 2, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels1),
            nn.Conv2d(
                out_channels1,
                out_channels1 * out_channels2,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, kernel_size // 2),
                groups=out_channels1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels1 * out_channels2),
            ChannelShuffle(out_channels1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolutions, the batch norms, and the shuffle.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: The output of shape
            ``[B, outs[0] * outs[1], H', W']``. For an odd
            ``kernel_size``, ``H'`` is ``ceil(H / stride)`` and ``W'`` is
            ``ceil(W / stride)``.

        """
        return self.conv(x)


class Stem(nn.Module):
    """Stem of MicroNet.

    The stem is a `SpatialSepConvSF` with a kernel size of ``3``,
    followed by an in-place ``ReLU6``.

    Example:
        >>> import torch
        >>> stem = Stem(3, stride=2, outs=(3, 2))
        >>> stem(torch.zeros(1, 3, 32, 32)).shape
        torch.Size([1, 6, 16, 16])

    """

    def __init__(
        self, in_channels: int, stride: int, outs: tuple[int, int] = (4, 4)
    ):
        r"""Initialize the convolution and the activation.

        Args:
            in_channels (int): The number of input channels.
            stride (int): The stride along the height and the width.
            outs (tuple[int, int]): The channel layout of the
                `SpatialSepConvSF`. The first value is the number of
                output channels of the :math:`3 \times 1` convolution and
                the number of groups after it. The second value is the
                channel multiplier. The stem has
                ``outs[0] * outs[1]`` output channels.

        """
        super().__init__()
        self.stem = nn.Sequential(
            SpatialSepConvSF(in_channels, outs, 3, stride), nn.ReLU6(True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolution and ``ReLU6``.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: The output of shape
            ``[B, outs[0] * outs[1], ceil(H / stride), ceil(W / stride)]``,
            with values in ``[0, 6]``.

        """
        return self.stem(x)


class DepthSpatialSepConv(nn.Module):
    r"""Factorized depthwise convolution that expands the channels.

    A :math:`k \times 1` convolution with ``in_channels`` groups
    multiplies the channels by ``expand[0]`` and applies the stride
    along the height. A :math:`1 \times k` convolution with one group
    for each of its input channels multiplies the channels by
    ``expand[1]`` and applies the stride along the width. A batch norm
    follows each convolution. The convolutions have no bias, and the
    block has no activation.

    Example:
        >>> import torch
        >>> conv = DepthSpatialSepConv(4, (2, 3), kernel_size=3, stride=2)
        >>> conv(torch.zeros(1, 4, 9, 9)).shape
        torch.Size([1, 24, 5, 5])

    """

    def __init__(
        self,
        in_channels: int,
        expand: tuple[int, int],
        kernel_size: int,
        stride: int,
    ):
        r"""Initialize the two depthwise convolutions.

        Args:
            in_channels (int): The number of input channels.
            expand (tuple[int, int]): The channel multipliers of the
                :math:`k \times 1` and the :math:`1 \times k`
                convolution. The block has
                ``in_channels * expand[0] * expand[1]`` output channels.
            kernel_size (int): The kernel size :math:`k`. The padding is
                ``kernel_size // 2``.
            stride (int): The stride along the height and the width.

        """
        super().__init__()
        exp1, exp2 = expand
        intermediate_channels = in_channels * exp1
        out_channels = in_channels * exp1 * exp2

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                intermediate_channels,
                (kernel_size, 1),
                (stride, 1),
                padding=(kernel_size // 2, 0),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(intermediate_channels),
            nn.Conv2d(
                intermediate_channels,
                out_channels,
                (1, kernel_size),
                (1, stride),
                padding=(0, kernel_size // 2),
                groups=intermediate_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two convolutions and their batch norms.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: The output of shape
            ``[B, in_channels * expand[0] * expand[1], H', W']``. For an
            odd ``kernel_size``, ``H'`` is ``ceil(H / stride)`` and
            ``W'`` is ``ceil(W / stride)``.

        """
        return self.conv(x)
