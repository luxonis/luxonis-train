"""The DDRNet backbone.

DDRNet keeps a high-resolution branch at 1/8 of the input size beside a
low-resolution branch. The two branches exchange features between the
stages.

"""

from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    ConvBlock,
    ResNetBlock,
    ResNetBottleneck,
    UpscaleOnline,
)

from .blocks import DAPPM, BasicDDRBackbone, make_layer


class DDRNet(BaseNode):
    r"""DDRNet backbone for semantic segmentation.

    DDRNet (Deep Dual-resolution Network) runs two branches. The stem,
    ``layer1``, and ``layer2`` of a `BasicDDRBackbone` bring the input to
    1/8 of its size. From there, a high-resolution branch keeps 1/8 of
    the input size. A low-resolution branch continues through ``layer3``,
    ``layer4``, and ``layer5`` down to 1/64. The two branches exchange
    features after each ``layer3`` stage and after ``layer4``. A `DAPPM`
    block pools the ``layer5`` features at several scales. The node
    upscales the result to 1/8 of the input size and adds it to the
    high-resolution branch.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, C, H, W\right]`

    Outputs:
        - ``features`` (``list[Tensor]``): :math:`\left[B, e \cdot hrc,
          H/8, W/8\right]`, preceded by :math:`\left[B, hrc, H/8,
          W/8\right]` when ``use_aux_heads`` is ``True``. :math:`hrc` is
          ``high_resolution_channels`` and :math:`e` is
          ``layer5_bottleneck_expansion``.

    References:
        - Source: Adapted from `Deci-AI/super-gradients
          <https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py>`_
          (Apache-2.0) and `ydhongHIT/DDRNet
          <https://github.com/ydhongHIT/DDRNet>`_ (MIT). Paper: `Deep
          Dual-resolution Networks for Real-time and Accurate Semantic
          Segmentation of Road Scenes <https://arxiv.org/abs/2101.06085>`_.
        - License: Apache-2.0

    Notes:
        ``H`` and ``W`` must be multiples of ``8``. Otherwise the
        tensors of the two branches differ in size, and the fusion fails.
        `SegmentationModel` attaches a `DDRNetSegmentationHead` to the
        last output. By default, it also attaches a second
        `DDRNetSegmentationHead` with its own weights to the first
        output, as an auxiliary head.

    Variants:
        - ``"23-slim"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``channels``: ``32``
                - ``high_resolution_channels``: ``64``
        - ``"23"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``channels``: ``64``
                - ``high_resolution_channels``: ``128``

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: DDRNet
              variant: 23-slim

    Compatible with:
        - Attach index: ``-1``, the last output of the input node
        - Used by: `SegmentationModel`
        - Pretrained weights: available through ``weights: download``

    """

    in_channels: int

    def __init__(
        self,
        channels: int = 32,
        high_resolution_channels: int = 64,
        use_aux_heads: bool = True,
        upscale_module: nn.Module | None = None,
        spp_width: int = 128,
        ssp_interpolation_mode: str = "bilinear",
        segmentation_interpolation_mode: str = "bilinear",
        # TODO: nn.Module registry
        block: type[nn.Module] = ResNetBlock,
        skip_block: type[nn.Module] = ResNetBlock,
        layer5_block: type[nn.Module] = ResNetBottleneck,
        layer5_bottleneck_expansion: int = 2,
        spp_kernel_sizes: list[int] | None = None,
        spp_strides: list[int] | None = None,
        layer3_repeats: int = 1,
        layers: list[int] | None = None,
        **kwargs,
    ):
        """Build the branches, the fusion layers, and the DAPPM block.

        The constructor reads `BaseNode.in_channels`, so the call must
        give ``input_shapes`` or ``in_sizes``. The class annotation
        ``in_channels: int`` makes `BaseNode` raise `IncompatibleError`
        when the attached input is a list of sizes, for example with
        ``attach_index="all"``. The constructor calls
        `BasicDDRBackbone.get_backbone_output_number_of_channels` to read
        the channel counts of ``layer2``, ``layer3``, and ``layer4``. That
        call draws random numbers and updates the running statistics of
        the backbone batch norms. `initialize_weights` does not reset
        these statistics.

        Args:
            channels (int): Number of stem channels of the backbone.
                ``layer2``, ``layer3``, and ``layer4`` have 2, 4, and 8
                times as many channels. A selected variant sets it,
                unless the call gives it explicitly.
            high_resolution_channels (int): Number of channels of the
                high-resolution branch. A selected variant sets it,
                unless the call gives it explicitly.
            use_aux_heads (bool): Whether `forward` also returns the
                high-resolution features after the last ``layer3``
                fusion, for an auxiliary head. Defaults to ``True``.
            upscale_module (``nn.Module | None``): Module that resizes
                the low-resolution features to 1/8 of the input size. The
                node calls it as ``upscale_module(x, height, width)``.
                ``None`` selects `UpscaleOnline` in the ``"bilinear"``
                mode.
            spp_width (int): Number of output channels of each `DAPPM`
                branch. Defaults to ``128``.
            ssp_interpolation_mode (str): Interpolation mode of the
                `DAPPM` branches. Defaults to ``"bilinear"``.
            segmentation_interpolation_mode (str): Value of the attribute
                ``segmentation_interpolation_mode``. The node does not
                use it. Defaults to ``"bilinear"``.
            block (``type[nn.Module]``): Block class of ``layer1`` to
                ``layer4`` in the `BasicDDRBackbone`. Defaults to
                `ResNetBlock`.
            skip_block (``type[nn.Module]``): Block class of the
                high-resolution stages ``layer3_skip`` and
                ``layer4_skip``. Defaults to `ResNetBlock`.
            layer5_block (``type[nn.Module]``): Block class of
                ``layer5`` and ``layer5_skip``. Defaults to
                `ResNetBottleneck`.
            layer5_bottleneck_expansion (int): Expansion factor of the
                ``layer5`` and ``layer5_skip`` blocks. The final output
                has ``high_resolution_channels * layer5_bottleneck_expansion``
                channels. Defaults to ``2``.
            spp_kernel_sizes (list[int] | None): Kernel size of each
                `DAPPM` branch. It must have the length of
                ``spp_strides``. Otherwise, `DAPPM` raises ``ValueError``.
                ``None`` or an empty list selects ``[1, 5, 9, 17, 0]``.
            spp_strides (list[int] | None): Stride of each `DAPPM`
                branch. ``None`` or an empty list selects
                ``[1, 2, 4, 8, 0]``.
            layer3_repeats (int): Number of ``layer3`` stages. A fusion
                of the two branches follows each stage. With a value
                below ``1``, `forward` skips ``layer3``. ``layer4`` then
                gets the wrong number of channels, and `forward` fails.
                Defaults to ``1``.
            layers (list[int] | None): Number of blocks in each stage, as
                eight entries: ``layer1``, ``layer2``, ``layer3``,
                ``layer4``, ``layer5``, ``layer3_skip``, ``layer4_skip``,
                and ``layer5_skip``. The ``layer3`` and ``layer3_skip``
                entries apply to each of the ``layer3_repeats`` stages.
                ``None`` or an empty list selects
                ``[2, 2, 2, 2, 1, 2, 2, 1]``.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseNode`.

        """
        super().__init__(**kwargs)

        upscale_module = upscale_module or UpscaleOnline()
        spp_kernel_sizes = spp_kernel_sizes or [1, 5, 9, 17, 0]
        spp_strides = spp_strides or [1, 2, 4, 8, 0]
        layers = layers or [2, 2, 2, 2, 1, 2, 2, 1]

        self._use_aux_heads = use_aux_heads
        self.upscale = upscale_module
        self._ssp_interpolation_mode = ssp_interpolation_mode
        self._segmentation_interpolation_mode = segmentation_interpolation_mode
        self.relu = nn.ReLU(inplace=False)
        self._layer3_repeats = layer3_repeats
        self._channels = channels
        self._layers = layers
        self._backbone_layers, self._additional_layers = (
            self._layers[:4],
            self._layers[4:],
        )

        self.backbone = BasicDDRBackbone(
            block=block,
            stem_channels=self._channels,
            layers=self._backbone_layers,
            in_channels=self.in_channels,
            layer3_repeats=self._layer3_repeats,
        )
        out_chan_backbone = (
            self.backbone.get_backbone_output_number_of_channels()
        )

        # Define layers for layer 3
        self.compression3 = nn.ModuleList()
        self.down3 = nn.ModuleList()
        self.layer3_skip = nn.ModuleList()
        for i in range(layer3_repeats):
            self.compression3.append(
                ConvBlock(
                    in_channels=out_chan_backbone["layer3"],
                    out_channels=high_resolution_channels,
                    kernel_size=1,
                    bias=False,
                    activation=None,
                )
            )
            self.down3.append(
                ConvBlock(
                    in_channels=high_resolution_channels,
                    out_channels=out_chan_backbone["layer3"],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    activation=None,
                )
            )
            self.layer3_skip.append(
                make_layer(
                    in_channels=(
                        out_chan_backbone["layer2"]
                        if i == 0
                        else high_resolution_channels
                    ),
                    channels=high_resolution_channels,
                    block=skip_block,
                    n_blocks=self._additional_layers[1],
                )
            )

        self.compression4 = ConvBlock(
            in_channels=out_chan_backbone["layer4"],
            out_channels=high_resolution_channels,
            kernel_size=1,
            bias=False,
            activation=None,
        )

        self.down4 = nn.Sequential(
            ConvBlock(
                in_channels=high_resolution_channels,
                out_channels=high_resolution_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                activation=nn.ReLU(inplace=True),
            ),
            ConvBlock(
                in_channels=high_resolution_channels * 2,
                out_channels=out_chan_backbone["layer4"],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                activation=None,
            ),
        )

        self.layer4_skip = make_layer(
            block=skip_block,
            in_channels=high_resolution_channels,
            channels=high_resolution_channels,
            n_blocks=self._additional_layers[2],
        )
        self.layer5_skip = make_layer(
            block=layer5_block,
            in_channels=high_resolution_channels,
            channels=high_resolution_channels,
            n_blocks=self._additional_layers[3],
            expansion=layer5_bottleneck_expansion,
        )

        self.layer5 = make_layer(
            block=layer5_block,
            in_channels=out_chan_backbone["layer4"],
            channels=out_chan_backbone["layer4"],
            n_blocks=self._additional_layers[0],
            stride=2,
            expansion=layer5_bottleneck_expansion,
        )

        self.spp = DAPPM(
            in_channels=out_chan_backbone["layer4"]
            * layer5_bottleneck_expansion,
            branch_channels=spp_width,
            out_channels=high_resolution_channels
            * layer5_bottleneck_expansion,
            interpolation_mode=self._ssp_interpolation_mode,
            kernel_sizes=spp_kernel_sizes,
            strides=spp_strides,
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        """Run both branches and return the high-resolution features.

        The stem, ``layer1``, and ``layer2`` bring the input to 1/8 of
        its size. The high-resolution branch starts there. It runs
        ``layer3_skip``, ``layer4_skip``, and ``layer5_skip`` beside
        ``layer3``, ``layer4``, and ``layer5`` of the low-resolution
        branch. The branches fuse after each ``layer3`` stage and after
        ``layer4``:

        - Strided ``3x3`` convolutions reduce the high-resolution
          features to the size of the low-resolution branch and add
          them to that branch. One convolution runs after each
          ``layer3`` stage, and two run after ``layer4``.
        - A ``1x1`` convolution and ``upscale_module`` resize the
          low-resolution features to 1/8 of the input size and add them
          to the high-resolution branch.

        Last, `DAPPM` runs on the ``layer5`` output. The node upscales
        the result and adds it to the ``layer5_skip`` output. A ReLU runs
        before every stage after ``layer1`` and before every fusion
        convolution.

        Args:
            inputs (``Tensor``): Image batch of shape ``[B, C, H, W]``.
                ``H`` and ``W`` must be multiples of ``8``. Other sizes
                make the fusion fail.

        Returns:
            ``list[Tensor]``: ``[features]``. ``features`` has the shape
            ``[B, C_out, H / 8, W / 8]``, where ``C_out`` is
            ``high_resolution_channels * layer5_bottleneck_expansion``.
            When ``use_aux_heads`` is ``True``, the list is
            ``[aux_features, features]``. ``aux_features`` holds the
            high-resolution features after the last ``layer3`` fusion, of
            shape ``[B, high_resolution_channels, H / 8, W / 8]``.

        Example:
            The batch has two images. In the training state, a batch
            norm at the ``1x1`` size of ``layer5`` needs more than one
            value per channel.

            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import DDRNet
            >>> node = DDRNet(
            ...     variant="23-slim",
            ...     input_shapes=[{"features": [Size([2, 3, 64, 64])]}],
            ... )
            >>> [tuple(t.shape) for t in node(torch.zeros(2, 3, 64, 64))]
            [(2, 64, 8, 8), (2, 128, 8, 8)]

        """
        width_output = inputs.shape[-1] // 8
        height_output = inputs.shape[-2] // 8

        x = self.backbone.stem(inputs)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(self.relu(x))

        # Repeat layer 3
        x_skip = x
        for i in range(self._layer3_repeats):
            out_layer3 = self.backbone.layer3[i](self.relu(x))
            out_layer3_skip = self.layer3_skip[i](self.relu(x_skip))

            x = out_layer3 + self.down3[i](self.relu(out_layer3_skip))
            x_skip = out_layer3_skip + self.upscale(
                self.compression3[i](self.relu(out_layer3)),
                height_output,
                width_output,
            )

        # Save for auxiliary head
        if self._use_aux_heads:
            x_extra = x_skip

        out_layer4 = self.backbone.layer4(self.relu(x))
        out_layer4_skip = self.layer4_skip(self.relu(x_skip))

        x = out_layer4 + self.down4(self.relu(out_layer4_skip))
        x_skip = out_layer4_skip + self.upscale(
            self.compression4(self.relu(out_layer4)),
            height_output,
            width_output,
        )

        out_layer5_skip = self.layer5_skip(self.relu(x_skip))

        x = self.upscale(
            self.spp(self.layer5(self.relu(x))), height_output, width_output
        )

        x = x + out_layer5_skip

        if self._use_aux_heads:
            return [x_extra, x]
        return [x]

    @override
    def initialize_weights(self, method: str | None = None) -> None:
        """Initialize the convolutions and the batch norms of the node.

        Every `torch.nn.Conv2d` gets Kaiming normal weights with
        ``mode="fan_out"`` and ``nonlinearity="relu"``, and a zero bias
        when it has a bias. Every `torch.nn.BatchNorm2d` gets the weight
        ``1`` and the bias ``0``. The method does not change the running
        statistics of the batch norms.

        After construction, `BaseNode` calls the method with ``weights``
        as ``method``. It skips the call only when ``weights`` is
        ``"download"`` or contains ``"://"``. Thus the method also gets
        a local checkpoint path, and it does not load that checkpoint.

        Args:
            method (str | None): Not used. Every value gives the same
                initialization. The method does not call
                `BaseNode.initialize_weights`, so ``"yolo"`` has no
                effect.

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @override
    def get_weights_url(self) -> str:
        if self._variant is None:
            raise ValueError(
                f"Online weights are available for '{self.name}' "
                "only when it's used with a predefined variant."
            )
        variant = self._variant.replace("-", "")
        return f"{{github}}/ddrnet_{variant}_coco.ckpt"

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Return the default variant name and the variants of DDRNet.

        ``"23-slim"`` is the default. It sets ``channels`` to ``32`` and
        ``high_resolution_channels`` to ``64``. ``"23"`` doubles both
        values, to ``64`` and ``128``.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The name ``"23-slim"``, and
            a dictionary that maps each variant name to its constructor
            keyword arguments.

        Example:
            >>> from luxonis_train.nodes import DDRNet
            >>> default, variants = DDRNet.get_variants()
            >>> default, variants["23"]
            ('23-slim', {'channels': 64, 'high_resolution_channels': 128})

        """
        return "23-slim", {
            "23-slim": {
                "channels": 32,
                "high_resolution_channels": 64,
            },
            "23": {
                "channels": 64,
                "high_resolution_channels": 128,
            },
        }
