"""The convolution and attention blocks of the EfficientViT backbone."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typeguard import typechecked

from luxonis_train.nodes.blocks import ConvBlock, autopad


class DepthWiseSeparableConv(nn.Module):
    """Depthwise separable convolution with an optional residual
    connection.

    The block runs a depthwise `ConvBlock` and then a ``1x1`` pointwise
    `ConvBlock`. The depthwise convolution has one group for each input
    channel. Both convolutions have a batch norm.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.backbones.efficientvit.blocks import (
        ...     DepthWiseSeparableConv,
        ... )
        >>> block = DepthWiseSeparableConv(8, 16, stride=2)
        >>> block(torch.zeros(1, 8, 15, 15)).shape
        torch.Size([1, 16, 8, 8])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        depthwise_bias: bool = False,
        pointwise_bias: bool = False,
        depthwise_activation: nn.Module | None = None,
        pointwise_activation: nn.Module | None = None,
        padding: int | str | None = None,
        dilation: int | tuple[int, int] = 1,
        use_residual: bool = False,
    ):
        """Build the depthwise and the pointwise convolutions.

        Args:
            in_channels (int): Number of input channels. The depthwise
                convolution keeps this number of channels.
            out_channels (int): Number of output channels of the
                pointwise convolution.
            kernel_size (int): Kernel size of the depthwise convolution.
                Defaults to ``3``.
            stride (int): Stride of the depthwise convolution. Defaults
                to ``1``.
            depthwise_bias (bool): Whether the depthwise convolution has
                a bias term. Defaults to ``False``.
            pointwise_bias (bool): Whether the pointwise convolution has
                a bias term. Defaults to ``False``.
            depthwise_activation (``nn.Module | None``): The activation
                after the depthwise convolution. ``None`` selects
                `torch.nn.ReLU6`.
            pointwise_activation (``nn.Module | None``): The activation
                after the pointwise convolution. ``None`` selects no
                activation.
            padding (int | str | None): Padding of the depthwise
                convolution, or the string ``"same"`` or ``"valid"``.
                ``None`` selects ``kernel_size // 2``. This value keeps
                the size only for an odd kernel size, a stride of ``1``,
                and a dilation of ``1``.
            dilation (int | tuple[int, int]): Dilation of the depthwise
                convolution. Defaults to ``1``.
            use_residual (bool): Whether `forward` adds the input to the
                output. The input and the output must then have the same
                shape. Defaults to ``False``.

        """
        super().__init__()

        self._use_residual = use_residual

        self.depthwise_conv = ConvBlock(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding=autopad(kernel_size) if padding is None else padding,
            dilation=dilation,
            groups=in_channels,
            activation=depthwise_activation or nn.ReLU6(),
            bias=depthwise_bias,
        )
        self.pointwise_conv = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            activation=pointwise_activation,
            bias=pointwise_bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the depthwise and the pointwise convolutions.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H', W']``. With
            the default padding, an odd kernel size, and a dilation of
            ``1``, :math:`H' = \lceil H / s \rceil` and
            :math:`W' = \lceil W / s \rceil`, where :math:`s` is
            ``stride``. When ``use_residual`` is ``True``, the method adds
            the input to the result.

        """
        identity = x
        x = self.pointwise_conv(self.depthwise_conv(x))
        if self._use_residual:
            x = x + identity
        return x


class MobileBottleneckBlock(nn.Module):
    """Mobile inverted bottleneck block.

    The block runs three `ConvBlock` layers. A ``1x1`` convolution
    expands the channels. A depthwise convolution with one group for
    each hidden channel follows. A second ``1x1`` convolution projects
    the channels to ``out_channels``. Each of the three layers has its
    own bias, batch norm, and activation settings.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.backbones.efficientvit.blocks import (
        ...     MobileBottleneckBlock,
        ... )
        >>> block = MobileBottleneckBlock(8, 16, stride=2, expand_ratio=4)
        >>> block.depthwise_conv.conv.in_channels
        32
        >>> block(torch.zeros(1, 8, 16, 16)).shape
        torch.Size([1, 16, 8, 8])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: float = 6,
        use_bias: list[bool] | None = None,
        use_norm: list[bool] | None = None,
        activation: list[nn.Module] | None = None,
        use_residual: bool = False,
    ):
        """Build the expansion, the depthwise, and the projection
        layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of the depthwise convolution.
                The padding is ``kernel_size // 2``. Defaults to ``3``.
            stride (int): Stride of the depthwise convolution. Defaults
                to ``1``.
            expand_ratio (float): Channel expansion factor. The hidden
                layers have ``round(in_channels * expand_ratio)``
                channels. Defaults to ``6``.
            use_bias (list[bool] | None): Whether each layer has a bias
                term, as three values for the expansion, the depthwise,
                and the projection layers. ``None`` selects
                ``[False, False, False]``.
            use_norm (list[bool] | None): Whether each layer has a batch
                norm, in the same order. ``None`` selects
                ``[True, True, True]``.
            activation (``list[nn.Module] | None``): The activation after
                each layer, in the same order. ``None`` selects
                `torch.nn.ReLU6`, `torch.nn.ReLU6`, and
                `torch.nn.Identity`.
            use_residual (bool): Whether `forward` adds the input to the
                output. The input and the output must then have the same
                shape. Defaults to ``False``.

        """
        super().__init__()

        if use_bias is None:
            use_bias = [False, False, False]
        if use_norm is None:
            use_norm = [True, True, True]
        if activation is None:
            activation = [nn.ReLU6(), nn.ReLU6(), nn.Identity()]

        self._use_residual = use_residual
        mid_channels = round(in_channels * expand_ratio)

        self.expand_conv = ConvBlock(
            in_channels,
            mid_channels,
            1,
            stride=1,
            use_norm=use_norm[0],
            activation=activation[0],
            bias=use_bias[0],
        )
        self.depthwise_conv = ConvBlock(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            padding=autopad(kernel_size),
            groups=mid_channels,
            use_norm=use_norm[1],
            activation=activation[1],
            bias=use_bias[1],
        )
        self.project_conv = ConvBlock(
            mid_channels,
            out_channels,
            1,
            use_norm=use_norm[2],
            activation=activation[2],
            bias=use_bias[2],
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the expansion, the depthwise, and the projection layers.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H', W']``. For
            an odd kernel size, :math:`H' = \lceil H / s \rceil` and
            :math:`W' = \lceil W / s \rceil`, where :math:`s` is
            ``stride``. When ``use_residual`` is ``True``, the method adds
            the input to the result.

        """
        identity = x
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self._use_residual:
            x = x + identity
        return x


class EfficientViTBlock(nn.Module):
    """EfficientViT block of an attention part and a convolution part.

    A `LightweightMLABlock` mixes the features of all positions. A
    `MobileBottleneckBlock` then mixes the features of adjacent
    positions. Both parts add their input to their output, so the block
    keeps the shape of its input.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.backbones.efficientvit.blocks import (
        ...     EfficientViTBlock,
        ... )
        >>> block = EfficientViTBlock(32, head_dim=8)
        >>> block(torch.zeros(1, 32, 8, 8)).shape
        torch.Size([1, 32, 8, 8])

    """

    @typechecked
    def __init__(
        self,
        n_channels: int,
        attention_ratio: float = 1.0,
        head_dim: int = 32,
        expansion_factor: float = 4.0,
        aggregation_scales: tuple[int, ...] = (5,),
    ):
        """Build the attention part and the convolution part.

        The attention part is a `LightweightMLABlock` with a batch norm
        only after its projection. The convolution part is a
        `MobileBottleneckBlock` with a ``3x3`` kernel. Its expansion and
        depthwise layers have a bias term and a `torch.nn.Hardswish`
        activation. Only its projection layer has a batch norm.

        Args:
            n_channels (int): Number of input and output channels.
            attention_ratio (float): Factor for the number of attention
                heads. The attention part has
                ``int(n_channels // head_dim * attention_ratio)`` heads.
                The number of heads must be at least ``1``. With ``0``
                heads and at least one aggregation scale,
                `torch.nn.Conv2d` raises ``ValueError``. Defaults to
                ``1.0``.
            head_dim (int): Number of channels of the query, the key, and
                the value of each attention head. Defaults to ``32``.
            expansion_factor (float): Channel expansion factor of the
                convolution part. Defaults to ``4.0``.
            aggregation_scales (``tuple[int, ...]``): Kernel size of the
                depthwise convolution of each multi-scale aggregation
                branch of the attention part. The values must be odd.
                Defaults to ``(5,)``.

        """
        super().__init__()

        self.attention_module = LightweightMLABlock(
            input_channels=n_channels,
            output_channels=n_channels,
            head_ratio=attention_ratio,
            dimension=head_dim,
            use_norm=[False, True],
            scale_factors=aggregation_scales,
            use_residual=True,
        )

        self.feature_module = MobileBottleneckBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            expand_ratio=expansion_factor,
            use_bias=[True, True, False],
            use_norm=[False, False, True],
            activation=[nn.Hardswish(), nn.Hardswish(), nn.Identity()],
            use_residual=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the attention part and then the convolution part.

        Args:
            x (``Tensor``): Input of shape ``[B, n_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, n_channels, H, W]``.

        """
        return self.feature_module(self.attention_module(x))


class LightweightMLABlock(nn.Module):
    r"""Lightweight multi-scale linear attention block of EfficientViT.

    A ``1x1`` `ConvBlock` computes the queries, the keys, and the values
    of all heads as one ``qkv`` tensor. Each aggregation branch runs a
    depthwise convolution with a kernel size from ``scale_factors`` and
    a grouped ``1x1`` convolution on that tensor. The block concatenates
    the ``qkv`` tensor and the branch outputs. Each head of each scale
    then computes its own attention. For the query :math:`Q`, the key
    :math:`K`, and the value :math:`V` of one head, the output at the
    position :math:`j` is:

    .. math::

        O_j = \frac{\sum_i V_i \, \phi(K_i)^\top \phi(Q_j)}
            {\sum_i \phi(K_i)^\top \phi(Q_j) + \epsilon}

    The sums run over all positions :math:`i`. :math:`\phi` is
    ``kernel_activation``. A ``1x1`` `ConvBlock` projects the attention
    outputs of all scales to ``output_channels``. When ``use_residual``
    is ``True``, the block adds the input to the projection output.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.backbones.efficientvit.blocks import (
        ...     LightweightMLABlock,
        ... )
        >>> block = LightweightMLABlock(16, 24, use_residual=False)
        >>> block.qkv_layer.conv.out_channels
        48
        >>> block(torch.zeros(1, 16, 4, 4)).shape
        torch.Size([1, 24, 4, 4])

    """

    @typechecked
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        n_heads: int | None = None,
        head_ratio: float = 1.0,
        dimension: int = 8,
        use_bias: list[bool] | None = None,
        use_norm: list[bool] | None = None,
        activations: list[nn.Module] | None = None,
        scale_factors: tuple[int, ...] = (5,),
        epsilon: float = 1e-15,
        use_residual: bool = True,
        kernel_activation: nn.Module | None = None,
    ):
        r"""Build the ``qkv`` layer, the aggregation branches, and the
        projection.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels. It must be
                equal to ``input_channels`` when ``use_residual`` is
                ``True``.
            n_heads (int | None): Number of attention heads. ``None`` or
                ``0`` selects
                ``int(input_channels // dimension * head_ratio)``. The
                number of heads must be at least ``1``. With ``0`` heads
                and at least one aggregation branch, `torch.nn.Conv2d`
                raises ``ValueError``.
            head_ratio (float): Factor for the number of heads when
                ``n_heads`` is ``None`` or ``0``. Defaults to ``1.0``.
            dimension (int): Number of channels of the query, the key,
                and the value of each head. Defaults to ``8``.
            use_bias (list[bool] | None): Whether the layers have a bias
                term, as two values. The first value applies to the
                ``qkv`` layer and to the convolutions of the aggregation
                branches. The second value applies to the projection.
                ``None`` selects ``[False, False]``.
            use_norm (list[bool] | None): Whether the ``qkv`` layer and
                the projection have a batch norm, as two values. ``None``
                selects ``[False, True]``.
            activations (``list[nn.Module] | None``): The activations
                after the ``qkv`` layer and after the projection. ``None``
                selects two `torch.nn.Identity` modules.
            scale_factors (``tuple[int, ...]``): Kernel size of the
                depthwise convolution of each aggregation branch. The
                block has one branch for each value. The values must be
                odd. An even value changes the height and the width of
                the branch output, and `forward` fails. Defaults to
                ``(5,)``.
            epsilon (float): Value that the attention adds to its
                denominator. Defaults to ``1e-15``.
            use_residual (bool): Whether `forward` adds the input to the
                output. Defaults to ``True``.
            kernel_activation (``nn.Module | None``): The kernel function
                :math:`\phi` that runs on the queries and the keys.
                ``None`` selects `torch.nn.ReLU`.

        """
        super().__init__()

        if use_bias is None:
            use_bias = [False, False]
        if use_norm is None:
            use_norm = [False, True]
        if activations is None:
            activations = [nn.Identity(), nn.Identity()]
        if kernel_activation is None:
            kernel_activation = nn.ReLU()

        self._epsilon = epsilon
        self._use_residual = use_residual
        n_heads = n_heads or int(input_channels // dimension * head_ratio)

        total_dim = n_heads * dimension

        self._dimension = dimension
        self.qkv_layer = ConvBlock(
            input_channels,
            3 * total_dim,
            kernel_size=1,
            bias=use_bias[0],
            use_norm=use_norm[0],
            activation=activations[0],
        )

        self.multi_scale_aggregators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        kernel_size=scale,
                        padding=autopad(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        kernel_size=1,
                        groups=3 * n_heads,
                        bias=use_bias[0],
                    ),
                )
                for scale in scale_factors
            ]
        )

        self.kernel_activation = kernel_activation

        self.projection_layer = ConvBlock(
            total_dim * (1 + len(scale_factors)),
            output_channels,
            kernel_size=1,
            bias=use_bias[1],
            use_norm=use_norm[1],
            activation=activations[1],
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def linear_attention(self, qkv_tensor: Tensor) -> Tensor:
        r"""Compute the attention of each head at a linear cost.

        The method splits the channels of ``qkv_tensor`` into groups of
        ``3 * dimension`` channels. Each group holds the query, the key,
        and the value of one head. The method computes the attention
        formula that `LightweightMLABlock` shows. It pads :math:`V` with
        a row of ones and multiplies the result with
        :math:`\phi(K)^\top` first. The extra row then gives the
        denominator. Thus the cost grows linearly with ``H * W``.

        The method runs with CUDA autocast disabled. A ``float16`` input
        runs in ``float32``. A ``bfloat16`` input runs in ``bfloat16``,
        and the method converts the result to ``float32`` before the
        division.

        Args:
            qkv_tensor (``Tensor``): Queries, keys, and values of shape
                ``[B, G * 3 * dimension, H, W]``, where ``G`` is the
                number of groups.

        Returns:
            ``Tensor``: Attention output of shape
            ``[B, G * dimension, H, W]``. It is ``float32`` when the input
            is ``float16`` or ``bfloat16``.

        Example:
            The linear and the quadratic attention give the same values.

            >>> import torch
            >>> from luxonis_train.nodes.backbones.efficientvit.blocks import (
            ...     LightweightMLABlock,
            ... )
            >>> block = LightweightMLABlock(8, 8, dimension=4)
            >>> qkv = torch.arange(96.0).reshape(1, 24, 2, 2) / 96
            >>> linear = block.linear_attention(qkv)
            >>> linear.shape
            torch.Size([1, 8, 2, 2])
            >>> torch.allclose(linear, block.quadratic_attention(qkv))
            True

        """
        batch, _, height, width = qkv_tensor.size()

        if qkv_tensor.dtype == torch.float16:
            qkv_tensor = qkv_tensor.float()

        qkv_tensor = qkv_tensor.reshape(
            batch, -1, 3 * self._dimension, height * width
        )
        query, key, value = (
            qkv_tensor[:, :, : self._dimension],
            qkv_tensor[:, :, self._dimension : 2 * self._dimension],
            qkv_tensor[:, :, 2 * self._dimension :],
        )

        query = self.kernel_activation(query)
        key = self.kernel_activation(key)

        key_transpose = key.transpose(-1, -2)

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        output = value @ key_transpose @ query

        if output.dtype == torch.bfloat16:
            output = output.float()

        output = output[:, :, :-1] / (output[:, :, -1:] + self._epsilon)
        return output.reshape(batch, -1, height, width)

    @torch.autocast(device_type="cuda", enabled=False)
    def quadratic_attention(self, qkv_tensor: Tensor) -> Tensor:
        r"""Compute the attention of each head with a full attention map.

        The method splits the channels of ``qkv_tensor`` into groups as
        `linear_attention` does. It builds the attention map
        :math:`A_{ij} = \phi(K_i)^\top \phi(Q_j)` of shape
        ``[B, G, H * W, H * W]``. It divides each column :math:`j` of the
        map by :math:`\sum_i A_{ij} + \epsilon`. Then it returns
        :math:`O_j = \sum_i V_i A_{ij}` with the divided map. The values
        are the same as the values of `linear_attention`, but the cost
        grows with the square of ``H * W``.

        The method runs with CUDA autocast disabled. It normalizes a
        ``float16`` or ``bfloat16`` attention map in ``float32`` and then
        converts the map back to the input type.

        Args:
            qkv_tensor (``Tensor``): Queries, keys, and values of shape
                ``[B, G * 3 * dimension, H, W]``, where ``G`` is the
                number of groups.

        Returns:
            ``Tensor``: Attention output of shape
            ``[B, G * dimension, H, W]``, with the type of the input.

        """
        batch, _, height, width = qkv_tensor.size()

        qkv_tensor = qkv_tensor.reshape(
            batch, -1, 3 * self._dimension, height * width
        )
        query, key, value = (
            qkv_tensor[:, :, : self._dimension],
            qkv_tensor[:, :, self._dimension : 2 * self._dimension],
            qkv_tensor[:, :, 2 * self._dimension :],
        )

        query = self.kernel_activation(query)
        key = self.kernel_activation(key)

        attention_map = key.transpose(-1, -2) @ query
        original_dtype = attention_map.dtype

        if original_dtype in [torch.float16, torch.bfloat16]:
            attention_map = attention_map.float()

        attention_map = attention_map / (
            torch.sum(attention_map, dim=2, keepdim=True) + self._epsilon
        )
        attention_map = attention_map.to(original_dtype)

        output = value @ attention_map
        return output.reshape(batch, -1, height, width)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the multi-scale attention and project the result.

        The method runs the ``qkv`` layer and the aggregation branches,
        and concatenates their outputs. It calls `linear_attention` when
        ``H * W`` is larger than ``dimension``. Otherwise, it calls
        `quadratic_attention`. It converts the result of
        `linear_attention` back to the type of the ``qkv`` tensor. The
        projection then maps the result to ``output_channels``. When
        ``use_residual`` is ``True``, the method adds the input to the
        projection output in place.

        Args:
            x (``Tensor``): Input of shape ``[B, input_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, output_channels, H, W]``.

        """
        identity = x
        qkv_output = self.qkv_layer(x)

        multi_scale_outputs = [qkv_output]
        multi_scale_outputs.extend(
            aggregator(qkv_output)
            for aggregator in self.multi_scale_aggregators
        )

        qkv_output = torch.cat(multi_scale_outputs, dim=1)

        height, width = qkv_output.size()[-2:]
        if height * width > self._dimension:
            attention_output = self.linear_attention(qkv_output).to(
                qkv_output.dtype
            )
        else:
            attention_output = self.quadratic_attention(qkv_output)

        final_output = self.projection_layer(attention_output)

        if self._use_residual:
            final_output += identity

        return final_output
