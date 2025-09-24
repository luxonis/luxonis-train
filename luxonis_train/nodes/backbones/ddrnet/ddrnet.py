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
    """DDRNet backbone for semantic segmentation.

    Variants
    ========
    The variant determines the number of channels and high resolution channels.
    The following variants are available:
        - "23-slim" (default): channels=32, high_resolution_channels=64
        - "23": channels=64, high_resolution_channels=128
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
        """DDRNet backbone.

        @see: U{Adapted from <https://github.com/Deci-AI/super-gradients/blob/master/src
            /super_gradients/training/models/segmentation_models/ddrnet.py>}
        @see: U{Original code <https://github.com/ydhongHIT/DDRNet>}
        @see: U{Paper <https://arxiv.org/pdf/2101.06085.pdf>}
        @license: U{Apache License, Version 2.0 <https://github.com/Deci-AI/super-
            gradients/blob/master/LICENSE.md>}
        @type channels: int | None
        @param channels: Base number of channels. If provided, overrides the variant values.
        @type high_resolution_channels: int | None
        @param high_resolution_channels: Number of channels in the high resolution net.
            If provided, overrides the variant values.
        @type use_aux_heads: bool
        @param use_aux_heads: Whether to use auxiliary heads. Defaults to True.
        @type upscale_module: nn.Module
        @param upscale_module: Module for upscaling (e.g., bilinear interpolation).
            Defaults to UpscaleOnline().
        @type spp_width: int
        @param spp_width: Width of the branches in the SPP block. Defaults to 128.
        @type ssp_interpolation_mode: str
        @param ssp_interpolation_mode: Interpolation mode for the SPP block. Defaults to
            "bilinear".
        @type segmentation_interpolation_mode: str
        @param segmentation_interpolation_mode: Interpolation mode for the segmentation head.
            Defaults to "bilinear".
        @type block: type[nn.Module]
        @param block: type of block to use in the backbone. Defaults to
            ResNetBlock.
        @type skip_block: type[nn.Module]
        @param skip_block: type of block for skip connections. Defaults to
            ResNetBlock.
        @type layer5_block: type[nn.Module]
        @param layer5_block: type of block for layer5 and layer5_skip. Defaults to
            Bottleneck.
        @type layer5_bottleneck_expansion: int
        @param layer5_bottleneck_expansion: Expansion factor for Bottleneck block in
            layer5. Defaults to 2.
        @type spp_kernel_sizes: list[int]
        @param spp_kernel_sizes: Kernel sizes for the SPP module pooling. Defaults to
            [1, 5, 9, 17, 0].
        @type spp_strides: list[int]
        @param spp_strides: Strides for the SPP module pooling. Defaults to [1, 2, 4, 8,
            0].
        @type layer3_repeats: int
        @param layer3_repeats: Number of times to repeat the 3rd stage. Defaults to 1.
        @type layers: list[int]
        @param layers: Number of blocks in each layer of the backbone. Defaults to [2,
            2, 2, 2, 1, 2, 2, 1].
        """
        super().__init__(**kwargs)

        upscale_module = upscale_module or UpscaleOnline()
        spp_kernel_sizes = spp_kernel_sizes or [1, 5, 9, 17, 0]
        spp_strides = spp_strides or [1, 2, 4, 8, 0]
        layers = layers or [2, 2, 2, 2, 1, 2, 2, 1]

        self._use_aux_heads = use_aux_heads
        self.upscale = upscale_module
        self.ssp_interpolation_mode = ssp_interpolation_mode
        self.segmentation_interpolation_mode = segmentation_interpolation_mode
        self.relu = nn.ReLU(inplace=False)
        self.layer3_repeats = layer3_repeats
        self.channels = channels
        self.layers = layers
        self.backbone_layers, self.additional_layers = (
            self.layers[:4],
            self.layers[4:],
        )

        self.backbone = BasicDDRBackbone(
            block=block,
            stem_channels=self.channels,
            layers=self.backbone_layers,
            in_channels=self.in_channels,
            layer3_repeats=self.layer3_repeats,
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
                    n_blocks=self.additional_layers[1],
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
            n_blocks=self.additional_layers[2],
        )
        self.layer5_skip = make_layer(
            block=layer5_block,
            in_channels=high_resolution_channels,
            channels=high_resolution_channels,
            n_blocks=self.additional_layers[3],
            expansion=layer5_bottleneck_expansion,
        )

        self.layer5 = make_layer(
            block=layer5_block,
            in_channels=out_chan_backbone["layer4"],
            channels=out_chan_backbone["layer4"],
            n_blocks=self.additional_layers[0],
            stride=2,
            expansion=layer5_bottleneck_expansion,
        )

        self.spp = DAPPM(
            in_channels=out_chan_backbone["layer4"]
            * layer5_bottleneck_expansion,
            branch_channels=spp_width,
            out_channels=high_resolution_channels
            * layer5_bottleneck_expansion,
            interpolation_mode=self.ssp_interpolation_mode,
            kernel_sizes=spp_kernel_sizes,
            strides=spp_strides,
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        width_output = inputs.shape[-1] // 8
        height_output = inputs.shape[-2] // 8

        x = self.backbone.stem(inputs)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(self.relu(x))

        # Repeat layer 3
        x_skip = x
        for i in range(self.layer3_repeats):
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
