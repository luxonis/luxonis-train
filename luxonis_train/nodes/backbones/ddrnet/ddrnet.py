"""DDRNet backbone.

Adapted from: U{https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py}
Original source: U{https://github.com/ydhongHIT/DDRNet}
Paper: U{https://arxiv.org/pdf/2101.06085.pdf}
@license: U{https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md}
"""
from typing import Dict, Type

from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule

from .blocks import (
    DAPPM,
    BasicDDRBackBone,
    BasicResNetBlock,
    Bottleneck,
    UpscaleOnline,
    _make_layer,
)


class DDRNet(BaseNode[Tensor, list[Tensor]]):
    def __init__(
        self,
        use_aux_heads: bool = True,
        upscale_module: nn.Module = None,
        highres_planes: int = 64,
        spp_width: int = 128,
        ssp_inter_mode: str = "bilinear",
        segmentation_inter_mode: str = "bilinear",
        block: Type[nn.Module] = BasicResNetBlock,
        skip_block: Type[nn.Module] = BasicResNetBlock,
        layer5_block: Type[nn.Module] = Bottleneck,
        layer5_bottleneck_expansion: int = 2,
        spp_kernel_sizes: list[int] = None,
        spp_strides: list[int] = None,
        layer3_repeats: int = 1,
        planes: int = 32,
        layers: list[int] = None,
        input_channels: int = 3,
        **kwargs,
    ):
        """Initialize the DDRNet with specified parameters.

        @type use_aux_heads: bool
        @param use_aux_heads: Whether to use auxiliary heads. Defaults to True.
        @type upscale_module: nn.Module
        @param upscale_module: Module for upscaling (e.g., bilinear interpolation).
            Defaults to UpscaleOnline().
        @type highres_planes: int
        @param highres_planes: Number of channels in the high resolution net. Defaults
            to 64.
        @type spp_width: int
        @param spp_width: Width of the branches in the SPP block. Defaults to 128.
        @type ssp_inter_mode: str
        @param ssp_inter_mode: Interpolation mode for the SPP block. Defaults to
            "bilinear".
        @type segmentation_inter_mode: str
        @param segmentation_inter_mode: Interpolation mode for the segmentation head.
            Defaults to "bilinear".
        @type block: Type[nn.Module]
        @param block: Type of block to use in the backbone. Defaults to
            BasicResNetBlock.
        @type skip_block: Type[nn.Module]
        @param skip_block: Type of block for skip connections. Defaults to
            BasicResNetBlock.
        @type layer5_block: Type[nn.Module]
        @param layer5_block: Type of block for layer5 and layer5_skip. Defaults to
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
        @type planes: int
        @param planes: Base number of channels. Defaults to 32.
        @type layers: list[int]
        @param layers: Number of blocks in each layer of the backbone. Defaults to [2,
            2, 2, 2, 1, 2, 2, 1].
        @type input_channels: int
        @param input_channels: Number of input channels. Defaults to 3.
        @type kwargs: Any
        @param kwargs: Additional arguments to pass to L{BaseNode}.
        """

        if upscale_module is None:
            upscale_module = UpscaleOnline()
        if spp_kernel_sizes is None:
            spp_kernel_sizes = [1, 5, 9, 17, 0]
        if spp_strides is None:
            spp_strides = [1, 2, 4, 8, 0]
        if layers is None:
            layers = [2, 2, 2, 2, 1, 2, 2, 1]

        super().__init__(**kwargs)

        self._use_aux_heads = use_aux_heads
        self.upscale = upscale_module
        self.ssp_inter_mode = ssp_inter_mode
        self.segmentation_inter_mode = segmentation_inter_mode
        self.block = block
        self.skip_block = skip_block
        self.relu = nn.ReLU(inplace=False)
        self.layer3_repeats = layer3_repeats
        self.planes = planes
        self.layers = layers
        self.backbone_layers, self.additional_layers = self.layers[:4], self.layers[4:]
        self.input_channels = input_channels

        self._backbone = BasicDDRBackBone(
            block=self.block,
            width=self.planes,
            layers=self.backbone_layers,
            input_channels=self.input_channels,
            layer3_repeats=self.layer3_repeats,
        )
        self._backbone.validate_backbone_attributes()
        out_chan_backbone = self._backbone.get_backbone_output_number_of_channels()

        # Define layers for layer 3
        self.compression3 = nn.ModuleList()
        self.down3 = nn.ModuleList()
        self.layer3_skip = nn.ModuleList()
        for i in range(layer3_repeats):
            self.compression3.append(
                ConvModule(
                    in_channels=out_chan_backbone["layer3"],
                    out_channels=highres_planes,
                    kernel_size=1,
                    bias=False,
                    activation=nn.Identity(),
                )
            )
            self.down3.append(
                ConvModule(
                    in_channels=highres_planes,
                    out_channels=out_chan_backbone["layer3"],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    activation=nn.Identity(),
                )
            )
            self.layer3_skip.append(
                _make_layer(
                    in_planes=out_chan_backbone["layer2"] if i == 0 else highres_planes,
                    planes=highres_planes,
                    block=skip_block,
                    num_blocks=self.additional_layers[1],
                )
            )

        self.compression4 = ConvModule(
            in_channels=out_chan_backbone["layer4"],
            out_channels=highres_planes,
            kernel_size=1,
            bias=False,
            activation=nn.Identity(),
        )

        self.down4 = nn.Sequential(
            ConvModule(
                in_channels=highres_planes,
                out_channels=highres_planes * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                activation=nn.ReLU(inplace=True),
            ),
            ConvModule(
                in_channels=highres_planes * 2,
                out_channels=out_chan_backbone["layer4"],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                activation=nn.Identity(),
            ),
        )

        self.layer4_skip = _make_layer(
            block=skip_block,
            in_planes=highres_planes,
            planes=highres_planes,
            num_blocks=self.additional_layers[2],
        )
        self.layer5_skip = _make_layer(
            block=layer5_block,
            in_planes=highres_planes,
            planes=highres_planes,
            num_blocks=self.additional_layers[3],
            expansion=layer5_bottleneck_expansion,
        )

        self.layer5 = _make_layer(
            block=layer5_block,
            in_planes=out_chan_backbone["layer4"],
            planes=out_chan_backbone["layer4"],
            num_blocks=self.additional_layers[0],
            stride=2,
            expansion=layer5_bottleneck_expansion,
        )

        self.spp = DAPPM(
            in_planes=out_chan_backbone["layer4"] * layer5_bottleneck_expansion,
            branch_planes=spp_width,
            out_planes=highres_planes * layer5_bottleneck_expansion,
            inter_mode=self.ssp_inter_mode,
            kernel_sizes=spp_kernel_sizes,
            strides=spp_strides,
        )

        self.highres_planes = highres_planes
        self.layer5_bottleneck_expansion = layer5_bottleneck_expansion
        self.init_params()

    @property
    def backbone(self):
        """Create a fake backbone module to load backbone pre-trained weights."""
        return nn.Sequential(
            Dict(
                [
                    ("_backbone", self._backbone),
                    ("compression3", self.compression3),
                    ("compression4", self.compression4),
                    ("down3", self.down3),
                    ("down4", self.down4),
                    ("layer3_skip", self.layer3_skip),
                    ("layer4_skip", self.layer4_skip),
                    ("layer5_skip", self.layer5_skip),
                ]
            )
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self._backbone.stem(x)
        x = self._backbone.layer1(x)
        x = self._backbone.layer2(self.relu(x))

        # Repeat layer 3
        x_skip = x
        for i in range(self.layer3_repeats):
            out_layer3 = self._backbone.layer3[i](self.relu(x))
            out_layer3_skip = self.layer3_skip[i](self.relu(x_skip))

            x = out_layer3 + self.down3[i](self.relu(out_layer3_skip))
            x_skip = out_layer3_skip + self.upscale(
                self.compression3[i](self.relu(out_layer3)), height_output, width_output
            )

        # Save for auxiliary head
        if self._use_aux_heads:
            x_extra = x_skip

        out_layer4 = self._backbone.layer4(self.relu(x))
        out_layer4_skip = self.layer4_skip(self.relu(x_skip))

        x = out_layer4 + self.down4(self.relu(out_layer4_skip))
        x_skip = out_layer4_skip + self.upscale(
            self.compression4(self.relu(out_layer4)), height_output, width_output
        )

        out_layer5_skip = self.layer5_skip(self.relu(x_skip))

        x = self.upscale(
            self.spp(self.layer5(self.relu(x))), height_output, width_output
        )

        x = x + out_layer5_skip

        if self._use_aux_heads:
            return [x, x_extra]
        else:
            return [x]

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
