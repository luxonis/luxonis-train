"""The encoder and decoder blocks of a U-Net, with and without skip
connections.
"""

from typing import Literal

import torch
from torch import Tensor, nn
from typeguard import typechecked

from .blocks import ConvBlock, ConvStack
from .utils import forward_gather


class EncoderBlock(nn.Sequential):
    """Encoder step of an optional ``2x2`` max pool and a `ConvStack`.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.blocks import EncoderBlock
        >>> x = torch.zeros(1, 3, 16, 16)
        >>> EncoderBlock(3, 8, n_repeats=2)(x).shape
        torch.Size([1, 8, 8, 8])
        >>> EncoderBlock(3, 8, n_repeats=2, max_pool=False)(x).shape
        torch.Size([1, 8, 16, 16])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_repeats: int,
        *,
        max_pool: bool = True,
    ):
        """Build the pooling and the convolution stack.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            n_repeats (int): The number of ``3x3`` `ConvBlock` layers in
                the stack.
            max_pool (bool): Whether a ``2x2`` max pool with stride ``2``
                halves the size before the stack. Otherwise the step
                keeps the size.

        """
        super().__init__(
            nn.MaxPool2d(2) if max_pool else nn.Identity(),
            ConvStack(in_channels, out_channels, n_repeats=n_repeats),
        )


class SimpleEncoder(nn.Sequential):
    r"""Encoder of `EncoderBlock` steps that returns the last feature map.

    The encoder has one step for each entry of ``width_multipliers``,
    and one more step with the last entry again. A step with the entry
    :math:`m` has :math:`\lfloor m \cdot c \rfloor` output channels,
    where :math:`c` is ``base_hidden_channels``. The first step keeps
    the size. Each later step halves it and rounds down. With :math:`n`
    entries, the output is thus :math:`2^n` times smaller than the
    input. `RecSubNet` pairs the encoder with `SimpleDecoder`.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.blocks import (
        ...     SimpleDecoder,
        ...     SimpleEncoder,
        ... )
        >>> encoder = SimpleEncoder(3, 8, [1, 2])
        >>> len(encoder)
        3
        >>> features = encoder(torch.zeros(1, 3, 16, 16))
        >>> features.shape
        torch.Size([1, 16, 4, 4])
        >>> SimpleDecoder(8, 3, [1, 2])(features).shape
        torch.Size([1, 3, 16, 16])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        base_hidden_channels: int,
        width_multipliers: list[float],
        n_convolutions: int = 2,
    ):
        """Build the encoder steps.

        Args:
            in_channels (int): The number of input channels.
            base_hidden_channels (int): The base width. Each step
                multiplies it by its entry of ``width_multipliers``.
            width_multipliers (list[float]): The width factor of each
                step. An empty list makes the constructor raise
                ``IndexError``.
            n_convolutions (int): The number of ``3x3`` `ConvBlock`
                layers in each step.

        """
        blocks = []
        for i, width_multiplier in enumerate(
            [*width_multipliers, width_multipliers[-1]]
        ):
            out_channels = int(base_hidden_channels * width_multiplier)
            blocks.append(
                EncoderBlock(
                    in_channels,
                    out_channels,
                    max_pool=i > 0,
                    n_repeats=n_convolutions,
                )
            )
            in_channels = out_channels
        super().__init__(*blocks)


class UNetEncoder(SimpleEncoder):
    """`SimpleEncoder` that returns the feature map of every step.

    `UNetDecoder` takes the list and uses the maps as skip connections.
    `DiscSubNetHead` pairs the two.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.blocks import UNetEncoder
        >>> encoder = UNetEncoder(3, 8, [1, 2])
        >>> features = encoder(torch.zeros(1, 3, 16, 16))
        >>> [tuple(feature.shape) for feature in features]
        [(1, 8, 16, 16), (1, 16, 8, 8), (1, 16, 4, 4)]

    """

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run the steps in order and collect every output.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``list[Tensor]``: One feature map for each step, from the
            largest to the smallest. The map of step ``i`` has
            :math:`2^i` times smaller height and width than ``x``.

        """
        return forward_gather(x, self)


class BaseDecoderBlock(nn.Module):
    """Base class of a decoder step: an `UpBlock` and a `ConvStack`.

    The class defines no ``forward``. `SimpleDecoderBlock` and
    `UNetDecoderBlock` add it.

    Attributes:
        up (UpBlock): The block that upsamples the input.
        conv (ConvStack): The convolution stack after the upsampling.

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_in_channels: int,
        kernel_size: int,
        use_norm: bool,
        align_corners: bool,
        upsample_mode: Literal[
            "simple_upsample", "conv_upsample", "conv_transpose"
        ],
        n_repeats: int,
    ):
        """Build the upsampling block and the convolution stack.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels of the
                upsampling block and of the stack.
            conv_in_channels (int): The number of input channels of the
                stack.
            kernel_size (int): The kernel size of `UpBlock`. Only
                ``"conv_transpose"`` uses it.
            use_norm (bool): Whether the ``3x3`` `ConvBlock` of `UpBlock`
                has a batch norm. The stack always has batch norms.
            align_corners (bool): The ``align_corners`` option of the
                interpolation in `UpBlock`.
            upsample_mode (``Literal["simple_upsample", "conv_upsample", "conv_transpose"]``):
                The upsampling method of `UpBlock`. The factor is ``2``.
            n_repeats (int): The number of ``3x3`` `ConvBlock` layers in
                the stack.

        """
        super().__init__()
        self.up = UpBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            use_norm=use_norm,
            align_corners=align_corners,
            upsample_mode=upsample_mode,
        )
        self.conv = ConvStack(
            conv_in_channels, out_channels, n_repeats=n_repeats
        )


class SimpleDecoderBlock(BaseDecoderBlock):
    """Decoder step without a skip connection.

    The step upsamples the input by ``2`` and runs the `ConvStack` on
    the result.

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_norm: bool,
        align_corners: bool,
        upsample_mode: Literal[
            "simple_upsample", "conv_upsample", "conv_transpose"
        ],
        n_repeats: int,
    ):
        """Build the upsampling block and the convolution stack.

        The stack maps ``out_channels`` channels to ``out_channels``.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The kernel size of `UpBlock`. Only
                ``"conv_transpose"`` uses it.
            use_norm (bool): Whether the ``3x3`` `ConvBlock` of `UpBlock`
                has a batch norm.
            align_corners (bool): The ``align_corners`` option of the
                interpolation in `UpBlock`.
            upsample_mode (``Literal["simple_upsample", "conv_upsample", "conv_transpose"]``):
                The upsampling method of `UpBlock`.
            n_repeats (int): The number of ``3x3`` `ConvBlock` layers in
                the stack.

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_in_channels=out_channels,
            kernel_size=kernel_size,
            use_norm=use_norm,
            align_corners=align_corners,
            upsample_mode=upsample_mode,
            n_repeats=n_repeats,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Upsample the input and run the convolution stack.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: The output of shape ``[B, out_channels, 2H, 2W]``
            for the interpolation modes. `UpBlock` gives the size for
            ``"conv_transpose"``.

        """
        x = self.up(x)
        return self.conv(x)


class UNetDecoderBlock(BaseDecoderBlock):
    """Decoder step with a skip connection from the encoder.

    The step upsamples the input by ``2`` and concatenates the skip
    feature map along the channel axis. Then it runs the `ConvStack` on
    the result. The skip feature map must have ``in_channels`` channels
    and the size of the upsampled input.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.blocks import UNetDecoderBlock
        >>> block = UNetDecoderBlock(
        ...     in_channels=8,
        ...     out_channels=4,
        ...     kernel_size=3,
        ...     use_norm=True,
        ...     align_corners=True,
        ...     upsample_mode="simple_upsample",
        ...     n_repeats=1,
        ... )
        >>> x, skip_x = torch.zeros(1, 8, 4, 4), torch.zeros(1, 8, 8, 8)
        >>> block(x, skip_x).shape
        torch.Size([1, 4, 8, 8])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_norm: bool,
        align_corners: bool,
        upsample_mode: Literal[
            "simple_upsample", "conv_upsample", "conv_transpose"
        ],
        n_repeats: int,
    ):
        """Build the upsampling block and the convolution stack.

        The stack maps ``in_channels + out_channels`` channels to
        ``out_channels``.

        Args:
            in_channels (int): The number of channels of the input and
                of the skip feature map.
            out_channels (int): The number of output channels.
            kernel_size (int): The kernel size of `UpBlock`. Only
                ``"conv_transpose"`` uses it.
            use_norm (bool): Whether the ``3x3`` `ConvBlock` of `UpBlock`
                has a batch norm.
            align_corners (bool): The ``align_corners`` option of the
                interpolation in `UpBlock`.
            upsample_mode (``Literal["simple_upsample", "conv_upsample", "conv_transpose"]``):
                The upsampling method of `UpBlock`.
            n_repeats (int): The number of ``3x3`` `ConvBlock` layers in
                the stack.

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_in_channels=in_channels + out_channels,
            kernel_size=kernel_size,
            use_norm=use_norm,
            align_corners=align_corners,
            upsample_mode=upsample_mode,
            n_repeats=n_repeats,
        )

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        """Upsample ``x``, concatenate ``skip_x``, and run the stack.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.
            skip_x (``Tensor``): The encoder feature map of shape
                ``[B, in_channels, 2H, 2W]``.

        Returns:
            ``Tensor``: The output of shape ``[B, out_channels, 2H, 2W]``.

        """
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)


class BaseDecoder(nn.Module):
    """Base class of a decoder that mirrors `SimpleEncoder`.

    The decoder adds ``1`` in front of the encoder multipliers and walks
    the list backwards. Each pair of neighbour entries gives one decoder
    step. The step doubles the size and maps the channels of the first
    entry to the channels of the second. For ``[1, 2]``, the steps map
    ``2 * base_width`` channels to ``base_width``, then ``base_width``
    to ``base_width``. A ``3x3`` convolution then maps ``base_width``
    channels to ``out_channels``.

    Every step uses ``"simple_upsample"`` with ``align_corners=True``,
    and a batch norm in its `UpBlock`. The class defines no ``forward``.
    `SimpleDecoder` and `UNetDecoder` add it.

    Attributes:
        blocks (``nn.ModuleList``): The decoder steps, from the smallest
            size to the largest.
        final_conv (``nn.Conv2d``): The ``3x3`` output convolution.

    """

    @typechecked
    def __init__(
        self,
        base_width: int,
        out_channels: int,
        encoder_width_multipliers: list[float],
        n_convolutions: int,
        block: type[SimpleDecoderBlock | UNetDecoderBlock],
    ):
        """Build the decoder steps and the output convolution.

        Args:
            base_width (int): The ``base_hidden_channels`` of the encoder.
            out_channels (int): The number of output channels.
            encoder_width_multipliers (list[float]): The
                ``width_multipliers`` of the encoder.
            n_convolutions (int): The number of ``3x3`` `ConvBlock`
                layers in the stack of each step.
            block (type[SimpleDecoderBlock | UNetDecoderBlock]): The class
                of the decoder steps.

        """
        super().__init__()
        self.blocks = nn.ModuleList()

        width_multipliers = [1, *encoder_width_multipliers]
        width_multipliers.reverse()
        for i in range(len(width_multipliers) - 1):
            self.blocks.append(
                block(
                    int(width_multipliers[i] * base_width),
                    int(width_multipliers[i + 1] * base_width),
                    kernel_size=3,
                    use_norm=True,
                    align_corners=True,
                    upsample_mode="simple_upsample",
                    n_repeats=n_convolutions,
                )
            )

        self.final_conv = nn.Conv2d(
            base_width, out_channels, kernel_size=3, padding=1
        )


class SimpleDecoder(BaseDecoder):
    """Decoder of `SimpleDecoderBlock` steps, without skip connections.

    The decoder takes the output of a `SimpleEncoder` whose
    ``base_hidden_channels`` equals ``base_width`` and whose multipliers
    are the same. It restores the input size of the encoder when that
    size is divisible by :math:`2^n`, where :math:`n` is the number of
    multipliers. The example of `SimpleEncoder` shows the pair.

    """

    @typechecked
    def __init__(
        self,
        base_width: int,
        out_channels: int,
        encoder_width_multipliers: list[float],
        n_convolutions: int = 2,
    ):
        super().__init__(
            base_width=base_width,
            out_channels=out_channels,
            encoder_width_multipliers=encoder_width_multipliers,
            n_convolutions=n_convolutions,
            block=SimpleDecoderBlock,
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Run the decoder steps and the output convolution.

        Args:
            x (``Tensor``): The output of `SimpleEncoder`, of shape
                ``[B, C, h, w]``, with
                :math:`C = \lfloor m \cdot w_b \rfloor`. Here :math:`m`
                is the last multiplier and :math:`w_b` is
                ``base_width``.

        Returns:
            ``Tensor``: The output of shape
            ``[B, out_channels, h * 2^n, w * 2^n]``, where ``n`` is the
            number of multipliers.

        """
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)


class UNetDecoder(BaseDecoder):
    """Decoder of `UNetDecoderBlock` steps with skip connections.

    The decoder takes the feature maps of a `UNetEncoder` whose
    ``base_hidden_channels`` equals ``base_width`` and whose
    multipliers are the same. `forward` removes the last map from the
    list, as the example shows.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.blocks import UNetDecoder, UNetEncoder
        >>> encoder = UNetEncoder(3, 8, [1, 2])
        >>> decoder = UNetDecoder(8, 2, [1, 2])
        >>> features = encoder(torch.zeros(1, 3, 16, 16))
        >>> decoder(features).shape
        torch.Size([1, 2, 16, 16])
        >>> len(features)
        2

    """

    @typechecked
    def __init__(
        self,
        base_width: int,
        out_channels: int,
        encoder_width_multipliers: list[float],
        n_convolutions: int = 2,
    ):
        super().__init__(
            base_width=base_width,
            out_channels=out_channels,
            encoder_width_multipliers=encoder_width_multipliers,
            n_convolutions=n_convolutions,
            block=UNetDecoderBlock,
        )

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """Decode the feature maps of `UNetEncoder`.

        The method pops the smallest map from ``inputs`` and starts from
        it. Each step then takes the next remaining map as its skip
        connection, from the smallest to the largest.

        **Warning:** The method removes the last element of ``inputs``.

        Args:
            inputs (``list[Tensor]``): The feature maps of `UNetEncoder`,
                from the largest to the smallest. The list must hold one
                map more than the decoder has steps. Otherwise ``zip``
                raises ``ValueError``.

        Returns:
            ``Tensor``: The output of shape ``[B, out_channels, H, W]``,
            where ``H`` and ``W`` are the size of the first map.

        """
        x = inputs.pop()
        for block, skip_x in zip(self.blocks, reversed(inputs), strict=True):
            x = block(x, skip_x)
        return self.final_conv(x)


class UpBlock(nn.Sequential):
    r"""Upsampling by ``stride``, followed by a ``3x3`` `ConvBlock`.

    ``upsample_mode`` selects the upsampling:

    - ``"conv_transpose"``: a `torch.nn.ConvTranspose2d` from
      ``in_channels`` to ``out_channels``, with ``kernel_size``,
      ``stride``, and no padding. The output size is
      :math:`(H - 1) \cdot s + k`, with the stride :math:`s` and the
      kernel size :math:`k`.
    - ``"simple_upsample"``: a `torch.nn.Upsample` by ``stride``. The
      `ConvBlock` then maps ``in_channels`` to ``out_channels``.
    - ``"conv_upsample"``: the same `torch.nn.Upsample`, then a ``1x1``
      `torch.nn.Conv2d` from ``in_channels`` to ``out_channels``.

    The `ConvBlock` keeps the size and gives ``out_channels`` channels.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.blocks import UpBlock
        >>> x = torch.zeros(1, 8, 4, 4)
        >>> options = {
        ...     "kernel_size": 2,
        ...     "use_norm": True,
        ...     "align_corners": False,
        ... }
        >>> UpBlock(8, 4, "conv_transpose", **options)(x).shape
        torch.Size([1, 4, 8, 8])
        >>> UpBlock(8, 4, "conv_upsample", **options)(x).shape
        torch.Size([1, 4, 8, 8])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_mode: Literal[
            "simple_upsample", "conv_upsample", "conv_transpose"
        ],
        kernel_size: int,
        use_norm: bool,
        align_corners: bool,
        stride: int = 2,
        activation: nn.Module | bool | None = True,
        interpolation_mode: Literal[
            "nearest", "linear", "bilinear", "bicubic", "trilinear"
        ] = "bilinear",
    ):
        """Build the upsampling layers and the convolution block.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            upsample_mode (``Literal["simple_upsample", "conv_upsample", "conv_transpose"]``):
                The upsampling method. The class description lists the
                layers of each method.
            kernel_size (int): The kernel size of the transposed
                convolution. The other methods ignore it.
            use_norm (bool): Whether the `ConvBlock` has a batch norm.
            align_corners (bool): The ``align_corners`` option of
                `torch.nn.Upsample`. ``"conv_transpose"`` ignores it.
            stride (int): The upsampling factor.
            activation (``nn.Module | bool | None``): The activation of
                the `ConvBlock`. ``True`` selects `torch.nn.ReLU`.
                ``False`` or ``None`` selects `torch.nn.Identity`.
            interpolation_mode (``Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"]``):
                The mode of `torch.nn.Upsample`. ``"conv_transpose"``
                ignores it. **Warning:** only ``"bilinear"`` and
                ``"bicubic"`` work on a 4D input. ``"nearest"`` rejects
                any ``align_corners`` value. ``"linear"`` needs a 3D
                input, and ``"trilinear"`` needs a 5D input.

        """
        layers = []

        if upsample_mode == "conv_transpose":
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            in_channels = out_channels
        else:
            layers.append(
                nn.Upsample(
                    scale_factor=stride,
                    mode=interpolation_mode,
                    align_corners=align_corners,
                )
            )
            if upsample_mode == "conv_upsample":
                layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1)
                )
                in_channels = out_channels

        layers.append(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_norm=use_norm,
                activation=activation,
            )
        )

        super().__init__(*layers)
