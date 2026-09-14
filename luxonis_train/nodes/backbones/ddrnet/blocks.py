"""The blocks of the DDRNet backbone.

`BasicDDRBackbone` holds the stem and the stages ``layer1`` to
``layer4``. `DAPPM` pools the deepest features at several scales.
`make_layer` stacks residual blocks into one stage.

References:
    - Adapted from: `super-gradients <https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py>`_
    - Original code: `ydhongHIT/DDRNet <https://github.com/ydhongHIT/DDRNet>`_
    - Paper: `Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes <https://arxiv.org/pdf/2101.06085.pdf>`_
    - License: `Apache License 2.0 <https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md>`_

"""

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvBlock, UpscaleOnline


class DAPPMBranch(nn.Module):
    """One pooling branch of the `DAPPM` block.

    The branch applies a batch norm, an optional downscale, a ReLU, and a
    ``1x1`` convolution to ``branch_channels``. `UpscaleOnline` then
    resizes the result to the height and width of the input. The
    ``stride`` selects the downscale:

    - ``0``: a global average pool to ``1x1``.
    - ``1``: no downscale.
    - Above ``1``: a depthwise convolution with the kernel size
      ``kernel_size``, the stride ``stride``, and the padding
      ``stride``.

    Example:
        >>> import torch
        >>> branch = DAPPMBranch(8, kernel_size=5, stride=2, branch_channels=4)
        >>> branch(torch.zeros(1, 8, 16, 16)).shape
        torch.Size([1, 4, 16, 16])

    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        branch_channels: int,
        interpolation_mode: str = "bilinear",
    ):
        """Initialize the downscale layers and the upscale.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Kernel size of the depthwise convolution.
                The branch uses it only when ``stride`` is above ``1``.
            stride (int): Selects the downscale. ``0`` selects a global
                average pool, ``1`` selects no downscale, and a larger
                value selects a depthwise convolution with this stride.
            branch_channels (int): Number of output channels.
            interpolation_mode (str): Mode of
                `torch.nn.functional.interpolate` for the resize to the
                input size. Defaults to ``"bilinear"``.

        """
        super().__init__()

        down_list = []
        down_list.append(nn.BatchNorm2d(in_channels))
        if stride == 0:
            down_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif stride > 1:
            down_list.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=stride,
                    groups=in_channels,
                    bias=False,
                )
            )

        down_list.append(nn.ReLU(inplace=True))
        down_list.append(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        )

        self.down_scale = nn.Sequential(*down_list)
        self.up_scale = UpscaleOnline(interpolation_mode)

    def forward(self, x: Tensor) -> Tensor:
        """Run the branch on ``x`` and resize the output to ``[H, W]``.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, branch_channels, H, W]``.

        """
        h, w = x.shape[-2], x.shape[-1]
        out = self.down_scale(x)
        return self.up_scale(out, output_height=h, output_width=w)


class MergeDAPPMBranch(DAPPMBranch):
    """A `DAPPM` branch that merges the output of the previous branch.

    The branch computes the `DAPPMBranch` output and adds the output of
    the previous branch to it. A batch norm, a ReLU, and a ``3x3``
    convolution then process the sum. The convolution keeps
    ``branch_channels``.

    Example:
        >>> import torch
        >>> branch = MergeDAPPMBranch(
        ...     8, kernel_size=5, stride=2, branch_channels=4
        ... )
        >>> previous = torch.zeros(1, 4, 16, 16)
        >>> branch(torch.zeros(1, 8, 16, 16), previous).shape
        torch.Size([1, 4, 16, 16])

    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        branch_channels: int,
        interpolation_mode: str = "bilinear",
    ):
        """Initialize the branch layers and the merge layers.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Kernel size of the depthwise convolution.
                The branch uses it only when ``stride`` is above ``1``.
            stride (int): Selects the downscale. ``0`` selects a global
                average pool, ``1`` selects no downscale, and a larger
                value selects a depthwise convolution with this stride.
            branch_channels (int): Number of output channels. The output
                of the previous branch must have the same number.
            interpolation_mode (str): Mode of
                `torch.nn.functional.interpolate` for the resize to the
                input size. Defaults to ``"bilinear"``.

        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            in_channels=in_channels,
            branch_channels=branch_channels,
            interpolation_mode=interpolation_mode,
        )

        self.process = nn.Sequential(
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        """Run the branch on ``x`` and merge the previous branch output.

        Args:
            x (``Tensor``): Input of the `DAPPM` block, of shape
                ``[B, in_channels, H, W]``.
            skip_x (``Tensor``): Output of the previous branch, of shape
                ``[B, branch_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, branch_channels, H, W]``.

        """
        out = super().forward(x)
        return self.process(out + skip_x)


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module (DAPPM) of DDRNet.

    The block runs one `DAPPMBranch` and a chain of `MergeDAPPMBranch`
    layers on the same input. Each merge branch adds the output of the
    branch before it. The block concatenates the outputs of all branches
    along the channel axis. A batch norm, a ReLU, and a ``1x1``
    convolution compress them to ``out_channels``. A shortcut of a batch
    norm, a ReLU, and a ``1x1`` convolution maps the input to
    ``out_channels``. The output is the sum of the compressed map and the
    shortcut.

    Example:
        >>> import torch
        >>> spp = DAPPM(
        ...     in_channels=8,
        ...     branch_channels=4,
        ...     out_channels=16,
        ...     kernel_sizes=[1, 5, 9, 17, 0],
        ...     strides=[1, 2, 4, 8, 0],
        ... )
        >>> len(spp.branches)
        4
        >>> spp(torch.zeros(1, 8, 16, 16)).shape
        torch.Size([1, 16, 16, 16])

    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        strides: list[int],
        interpolation_mode: str = "bilinear",
    ):
        """Initialize the branches, the compression, and the shortcut.

        Args:
            in_channels (int): Number of input channels.
            branch_channels (int): Number of output channels of each
                branch.
            out_channels (int): Number of output channels.
            kernel_sizes (list[int]): Kernel size of each branch. The
                first entry configures the `DAPPMBranch`. Each other
                entry configures one `MergeDAPPMBranch`.
            strides (list[int]): Stride of each branch, in the order of
                ``kernel_sizes``. `DAPPMBranch` explains the values
                ``0``, ``1``, and larger.
            interpolation_mode (str): Mode of
                `torch.nn.functional.interpolate` for the resize in each
                branch. Defaults to ``"bilinear"``.

        Raises:
            IndexError: When ``kernel_sizes`` or ``strides`` is empty.
            ValueError: When ``kernel_sizes`` and ``strides`` have
                different lengths.

        """
        super().__init__()

        self.start_branch = DAPPMBranch(
            in_channels=in_channels,
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            branch_channels=branch_channels,
            interpolation_mode=interpolation_mode,
        )
        self.branches = nn.ModuleList(
            [
                MergeDAPPMBranch(
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    branch_channels=branch_channels,
                    interpolation_mode=interpolation_mode,
                )
                for kernel_size, stride in zip(
                    kernel_sizes[1:], strides[1:], strict=True
                )
            ]
        )

        compression_channels = branch_channels * (len(self.branches) + 1)
        self.compression = nn.Sequential(
            nn.BatchNorm2d(compression_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                compression_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pool ``x`` at every scale and fuse the results.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H, W]``.

        """
        x_list = [self.start_branch(x)]

        for i, branch in enumerate(self.branches):
            x_list.append(branch(x, x_list[i]))

        return self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)


class BasicDDRBackbone(nn.Module):
    """The stem and the ResNet-like stages of DDRNet.

    The stem is two ``3x3`` `ConvBlock` layers. Each layer has stride
    ``2``, a bias, a batch norm, and a ReLU. `make_layer` builds the
    stages from ``block``. This list gives the output channels and the
    output height of each part, for an input of height ``H``. The width
    follows the same scale.

    - ``stem``: ``stem_channels`` channels at ``H / 4``.
    - ``layer1``: ``stem_channels`` channels at ``H / 4``.
    - ``layer2``: ``2 * stem_channels`` channels at ``H / 8``.
    - ``layer3``: a `torch.nn.ModuleList` of
      ``max(layer3_repeats, 1)`` stages, with ``4 * stem_channels``
      channels at ``H / 16``. Only the first stage has stride ``2``.
    - ``layer4``: ``8 * stem_channels`` channels at ``H / 32``.

    The class does not implement ``forward``, so a call of the module
    raises ``NotImplementedError``. `DDRNet` calls the stem and the
    stages one by one, and puts its high-resolution branch beside
    ``layer3`` and ``layer4``.

    """

    def __init__(
        self,
        block: type[nn.Module],
        stem_channels: int,
        layers: list[int],
        in_channels: int,
        layer3_repeats: int = 1,
    ):
        """Build the stem and the four stages.

        Args:
            block (``type[nn.Module]``): Block class of every stage, for
                example `ResNetBlock`. `make_layer` lists the arguments
                that the class must accept.
            stem_channels (int): Number of output channels of the stem
                and of ``layer1``. ``layer2``, ``layer3``, and ``layer4``
                have 2, 4, and 8 times as many channels.
            layers (list[int]): Number of blocks in ``layer1``,
                ``layer2``, each ``layer3`` stage, and ``layer4``. The
                backbone reads the first four entries.
            in_channels (int): Number of input channels.
            layer3_repeats (int): Number of ``layer3`` stages. A value
                below ``1`` still builds one stage. Defaults to ``1``.

        """
        super().__init__()
        self.input_channels = in_channels

        self.stem = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            ConvBlock(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
        )

        self.layer1 = make_layer(
            block=block,
            in_channels=stem_channels,
            channels=stem_channels,
            n_blocks=layers[0],
        )

        self.layer2 = make_layer(
            block=block,
            in_channels=stem_channels,
            channels=stem_channels * 2,
            n_blocks=layers[1],
            stride=2,
        )

        self.layer3 = nn.ModuleList(
            [
                make_layer(
                    block=block,
                    in_channels=stem_channels * 2,
                    channels=stem_channels * 4,
                    n_blocks=layers[2],
                    stride=2,
                )
            ]
            + [
                make_layer(
                    block=block,
                    in_channels=stem_channels * 4,
                    channels=stem_channels * 4,
                    n_blocks=layers[2],
                    stride=1,
                )
                for _ in range(layer3_repeats - 1)
            ]
        )

        self.layer4 = make_layer(
            block=block,
            in_channels=stem_channels * 4,
            channels=stem_channels * 8,
            n_blocks=layers[3],
            stride=2,
        )

    def get_backbone_output_number_of_channels(self) -> dict[str, int]:
        """Return the number of output channels of the later stages.

        The method runs a random CPU tensor of shape
        ``[1, in_channels, 320, 320]`` through the stem and all stages,
        so the backbone must be on the CPU. It reads the channel
        dimension after ``layer2``, after the last ``layer3`` stage, and
        after ``layer4``. The run draws from the global random generator
        of PyTorch. In the training state, the run also updates the
        running statistics of the batch norms.

        Returns:
            dict[str, int]: The channel counts under the keys
            ``"layer2"``, ``"layer3"``, and ``"layer4"``.

        Example:
            >>> from luxonis_train.nodes.blocks import ResNetBlock
            >>> backbone = BasicDDRBackbone(
            ...     ResNetBlock,
            ...     stem_channels=8,
            ...     layers=[1, 1, 1, 1],
            ...     in_channels=3,
            ... )
            >>> backbone.get_backbone_output_number_of_channels()
            {'layer2': 16, 'layer3': 32, 'layer4': 64}

        """
        output_shapes = {}
        x = torch.randn(1, self.input_channels, 320, 320)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        output_shapes["layer2"] = x.shape[1]

        for layer in self.layer3:
            x = layer(x)
        output_shapes["layer3"] = x.shape[1]

        x = self.layer4(x)
        output_shapes["layer4"] = x.shape[1]

        return output_shapes


def make_layer(
    block: type[nn.Module],
    in_channels: int,
    channels: int,
    n_blocks: int,
    stride: int = 1,
    expansion: int = 1,
) -> nn.Sequential:
    """Stack ``n_blocks`` residual blocks into one stage.

    The function calls ``block`` once for each block. Each call passes
    the input channels and ``channels`` as positional arguments, and
    ``stride``, ``final_relu``, and ``expansion`` as keyword arguments.
    The first block gets ``in_channels`` and ``stride``. The other
    blocks get ``channels * expansion`` input channels and the stride
    ``1``. The last block gets ``final_relu=False``, and every other
    block gets ``final_relu=True``. Thus the stage output has no final
    ReLU. A value of ``n_blocks`` below ``1`` still builds one block.

    Args:
        block (``type[nn.Module]``): Block class, for example
            `ResNetBlock` or `ResNetBottleneck`. It must accept the
            arguments above and output ``channels * expansion``
            channels.
        in_channels (int): Number of input channels of the first block.
        channels (int): Number of hidden channels of each block.
        n_blocks (int): Number of blocks.
        stride (int): Stride of the first block. Defaults to ``1``.
        expansion (int): Expansion factor of every block. Defaults to
            ``1``.

    Returns:
        ``nn.Sequential``: The blocks in order. The stage outputs
        ``channels * expansion`` channels.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.blocks import ResNetBlock
        >>> stage = make_layer(ResNetBlock, 8, 16, n_blocks=3, stride=2)
        >>> [type(block.final_relu).__name__ for block in stage]
        ['ReLU', 'ReLU', 'Identity']
        >>> stage(torch.zeros(1, 8, 8, 8)).shape
        torch.Size([1, 16, 4, 4])

    """
    layers: list[nn.Module] = []

    layers.append(
        block(
            in_channels,
            channels,
            stride=stride,
            final_relu=n_blocks > 1,
            expansion=expansion,
        )
    )

    in_channels = channels * expansion

    for i in range(1, n_blocks):
        final_relu = i != (n_blocks - 1)
        layers.append(
            block(
                in_channels,
                channels,
                stride=1,
                final_relu=final_relu,
                expansion=expansion,
            )
        )

    return nn.Sequential(*layers)
