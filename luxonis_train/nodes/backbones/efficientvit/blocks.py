import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typeguard import typechecked

from luxonis_train.nodes.blocks import ConvBlock, autopad


class DepthWiseSeparableConv(nn.Module):
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
        """Depthwise separable convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size. Defaults to 3.
            stride (int): Stride. Defaults to 1.
            depthwise_bias (bool): Whether to use bias for the depthwise convolution.
            pointwise_bias (bool): Whether to use bias for the pointwise convolution.
            depthwise_activation (nn.Module | None): Activation function for the depthwise convolution. Defaults to nn.ReLU6().
            pointwise_activation (nn.Module | None): Activation function for the pointwise convolution.
            padding (int | str | None): Padding. Defaults to None.
            dilation (int | tuple[int, int]): Dilation. Defaults to 1.
            use_residual (bool): Whether to add the input tensor to the output. Defaults to False.

        """
        super().__init__()

        self.use_residual = use_residual

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
        identity = x
        x = self.pointwise_conv(self.depthwise_conv(x))
        if self.use_residual:
            x = x + identity
        return x


class MobileBottleneckBlock(nn.Module):
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
        """EfficientViT mobile bottleneck block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size. Defaults to 3.
            stride (int): Stride. Defaults to 1.
            expand_ratio (float): Expansion ratio. Defaults to 6.
            use_bias (list[bool] | None): Whether to use bias for the depthwise and pointwise convolutions.
            use_norm (list[bool] | None): Whether to use normalization for the depthwise and pointwise convolutions.
            activation (list[nn.Module] | None): Activation functions for the depthwise and pointwise convolutions.
            use_residual (bool): Whether to use residual connection. Defaults to False.

        """
        super().__init__()

        if use_bias is None:
            use_bias = [False, False, False]
        if use_norm is None:
            use_norm = [True, True, True]
        if activation is None:
            activation = [nn.ReLU6(), nn.ReLU6(), nn.Identity()]

        self.use_residual = use_residual
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
        identity = x
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x = x + identity
        return x


class EfficientViTBlock(nn.Module):
    @typechecked
    def __init__(
        self,
        n_channels: int,
        attention_ratio: float = 1.0,
        head_dim: int = 32,
        expansion_factor: float = 4.0,
        aggregation_scales: tuple[int, ...] = (5,),
    ):
        """EfficientViT block for multi-scale linear attention and local
        features.

        Args:
            n_channels (int): The number of input and output channels.
            attention_ratio (float): Ratio for determining the number of attention heads. Default is 1.0.
            head_dim (int): Dimension size for each attention head. Default is 32.
            expansion_factor (float): Factor by which channels expand in the local module. Default is 4.0.
            aggregation_scales (tuple[int, ...]): Tuple defining the scales for aggregation in the attention module. Default is (5,).

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
        """Forward pass of the block.

        Args:
            x (Tensor): Input tensor with shape [batch, channels, height, width].

        Returns:
            Tensor: Output tensor after attention and local feature processing.

        """
        return self.feature_module(self.attention_module(x))


class LightweightMLABlock(nn.Module):
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
        """Efficient multi-scale linear attention block.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            n_heads (int | None): Number of attention heads. Defaults to None.
            head_ratio (float): Ratio to determine the number of heads. Default is 1.0.
            dimension (int): Size of each head. Default is 8.
            use_bias (list[bool] | None): List specifying if bias is used in qkv and projection layers.
            use_norm (list[bool] | None): List specifying if normalization is applied in qkv and projection layers.
            activations (list[nn.Module] | None): List of activation functions for qkv and projection layers.
            scale_factors (tuple[int, ...]): Tuple defining scales for aggregation. Default is (5,).
            epsilon (float): Epsilon value for numerical stability. Default is 1e-15.
            use_residual (bool): Whether to add the input tensor to the output. Defaults to True.
            kernel_activation (nn.Module | None): Activation used for attention kernels. Defaults to nn.ReLU().

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

        self.epsilon = epsilon
        self.use_residual = use_residual
        n_heads = n_heads or int(input_channels // dimension * head_ratio)

        total_dim = n_heads * dimension

        self.dimension = dimension
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
        """ReLU-based linear attention."""
        batch, _, height, width = qkv_tensor.size()

        if qkv_tensor.dtype == torch.float16:
            qkv_tensor = qkv_tensor.float()

        qkv_tensor = qkv_tensor.reshape(
            batch, -1, 3 * self.dimension, height * width
        )
        query, key, value = (
            qkv_tensor[:, :, : self.dimension],
            qkv_tensor[:, :, self.dimension : 2 * self.dimension],
            qkv_tensor[:, :, 2 * self.dimension :],
        )

        query = self.kernel_activation(query)
        key = self.kernel_activation(key)

        key_transpose = key.transpose(-1, -2)

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        output = value @ key_transpose @ query

        if output.dtype == torch.bfloat16:
            output = output.float()

        output = output[:, :, :-1] / (output[:, :, -1:] + self.epsilon)
        return output.reshape(batch, -1, height, width)

    @torch.autocast(device_type="cuda", enabled=False)
    def quadratic_attention(self, qkv_tensor: Tensor) -> Tensor:
        """ReLU-based quadratic attention."""
        batch, _, height, width = qkv_tensor.size()

        qkv_tensor = qkv_tensor.reshape(
            batch, -1, 3 * self.dimension, height * width
        )
        query, key, value = (
            qkv_tensor[:, :, : self.dimension],
            qkv_tensor[:, :, self.dimension : 2 * self.dimension],
            qkv_tensor[:, :, 2 * self.dimension :],
        )

        query = self.kernel_activation(query)
        key = self.kernel_activation(key)

        attention_map = key.transpose(-1, -2) @ query
        original_dtype = attention_map.dtype

        if original_dtype in [torch.float16, torch.bfloat16]:
            attention_map = attention_map.float()

        attention_map = attention_map / (
            torch.sum(attention_map, dim=2, keepdim=True) + self.epsilon
        )
        attention_map = attention_map.to(original_dtype)

        output = value @ attention_map
        return output.reshape(batch, -1, height, width)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        qkv_output = self.qkv_layer(x)

        multi_scale_outputs = [qkv_output]
        for aggregator in self.multi_scale_aggregators:
            multi_scale_outputs.append(aggregator(qkv_output))

        qkv_output = torch.cat(multi_scale_outputs, dim=1)

        height, width = qkv_output.size()[-2:]
        if height * width > self.dimension:
            attention_output = self.linear_attention(qkv_output).to(
                qkv_output.dtype
            )
        else:
            attention_output = self.quadratic_attention(qkv_output)

        final_output = self.projection_layer(attention_output)

        if self.use_residual:
            final_output += identity

        return final_output
