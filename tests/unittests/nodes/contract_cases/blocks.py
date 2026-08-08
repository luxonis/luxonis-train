"""Executable shape contract cases for `luxonis_train.nodes.blocks`."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from luxonis_train.nodes.blocks.blocks import (
    DFL,
    AttentionRefinmentBlock,
    BlockRepeater,
    BottleRep,
    ConvBlock,
    ConvStack,
    CSPStackRepBlock,
    DropPath,
    EfficientDecoupledBlock,
    FeatureFusionBlock,
    GeneralReparametrizableBlock,
    PreciseDecoupledBlock,
    SegProto,
    SpatialPyramidPoolingBlock,
    SqueezeExciteBlock,
    UpscaleOnline,
)
from luxonis_train.nodes.blocks.resnet import (
    GenericResidualBlock,
    ResNetBlock,
    ResNetBottleneck,
)
from luxonis_train.nodes.blocks.unet import (
    EncoderBlock,
    SimpleDecoder,
    SimpleDecoderBlock,
    SimpleEncoder,
    UNetDecoder,
    UNetDecoderBlock,
    UNetEncoder,
    UpBlock,
)

from .case import ContractCase, case, feature_map

UpsampleMode = Literal["simple_upsample", "conv_upsample", "conv_transpose"]

DECODER_BLOCK_KWARGS = {
    "kernel_size": 3,
    "use_norm": True,
    "align_corners": True,
    "upsample_mode": "simple_upsample",
    "n_repeats": 2,
}


def _general_reparametrizable_block() -> list[ContractCase]:
    fused = GeneralReparametrizableBlock(8, 8, n_branches=2)
    fused.reparametrize()
    return [
        feature_map(
            GeneralReparametrizableBlock(8, 8, n_branches=2),
            torch.randn(2, 8, 12, 12),
        ),
        feature_map(fused, torch.randn(2, 8, 12, 12)),
        feature_map(
            GeneralReparametrizableBlock(8, 16, stride=2, refine_block="se"),
            torch.randn(2, 8, 12, 12),
            out_channels=16,
            out_size=(6, 6),
        ),
    ]


def _decoupled(module: nn.Module, **extra: int) -> ContractCase:
    """Build a case for a block splitting features into three heads.

    These blocks have no single output, so the feature-map symbols do
    not apply to them.
    """
    x = torch.randn(2, 8, 10, 10)
    return ContractCase(
        module=module.eval(),
        inputs={"x": x},
        bindings={"B": 2, "C_in": 8, "H": 10, "W": 10, **extra},
    )


def _feature_fusion_block() -> ContractCase:
    return case(
        FeatureFusionBlock(12, 8),
        {
            "x1": torch.randn(2, 4, 10, 10),
            "x2": torch.randn(2, 8, 10, 10),
        },
        {"B": 2, "C1": 4, "C2": 8, "C_out": 8, "H": 10, "W": 10},
    )


def _upscale_online() -> ContractCase:
    return case(
        UpscaleOnline(),
        {
            "x": torch.randn(2, 6, 8, 8),
            "output_height": 13,
            "output_width": 11,
        },
        {
            "B": 2,
            "C_in": 6,
            "H": 8,
            "W": 8,
            "output_height": 13,
            "output_width": 11,
        },
    )


def _generic_residual_block() -> list[ContractCase]:
    return [
        feature_map(
            GenericResidualBlock(
                in_channels=8,
                hidden_channels=4,
                stride=1,
                expansion=2,
                final_relu=True,
                block=ConvBlock(8, 8, kernel_size=3, padding=1),
            ),
            torch.randn(2, 8, 12, 12),
        ),
        feature_map(
            GenericResidualBlock(
                in_channels=6,
                hidden_channels=4,
                stride=2,
                expansion=2,
                final_relu=False,
                block=ConvBlock(6, 8, kernel_size=3, stride=2, padding=1),
            ),
            torch.randn(2, 6, 12, 12),
            out_channels=8,
            out_size=(6, 6),
        ),
    ]


def _unet_encoder() -> ContractCase:
    return case(
        UNetEncoder(3, 4, [1.0, 2.0]),
        {"x": torch.randn(2, 3, 16, 16)},
        {
            "B": 2,
            "C_in": 3,
            "H_in": 16,
            "W_in": 16,
            "N": 3,
            "C": (4, 8, 8),
            "H": (16, 8, 4),
            "W": (16, 8, 4),
        },
        key="features",
    )


def _unet_decoder_block() -> ContractCase:
    return case(
        UNetDecoderBlock(8, 4, **DECODER_BLOCK_KWARGS),
        {
            "x": torch.randn(2, 8, 6, 6),
            "skip_x": torch.randn(2, 8, 12, 12),
        },
        {"B": 2, "C_in": 8, "C_skip": 8, "C_out": 4, "H": 6, "W": 6},
    )


def _unet_decoder() -> ContractCase:
    encoder = UNetEncoder(3, 4, [1.0, 2.0]).eval()
    features = encoder(torch.randn(2, 3, 16, 16))
    module = UNetDecoder(4, 2, [1.0, 2.0]).eval()
    # `UNetDecoder.forward` pops the deepest feature off the list it is
    # given, so it is run on a copy and the untouched features are what
    # the input side of the contract is checked against.
    output = module(list(features))
    return ContractCase(
        module=module,
        inputs={"inputs": features},
        bindings={
            "B": 2,
            "N": 3,
            "C": (4, 8, 8),
            "H": (16, 8, 4),
            "W": (16, 8, 4),
            "C_out": 2,
            "H_out": 16,
            "W_out": 16,
        },
        outputs={"output": output},
    )


def _up_block() -> list[ContractCase]:
    # `conv_transpose` grows an 8x8 map to 17x17, the interpolating modes
    # to 16x16.
    configurations: list[tuple[UpsampleMode, int]] = [
        ("simple_upsample", 16),
        ("conv_upsample", 16),
        ("conv_transpose", 17),
    ]
    return [
        feature_map(
            UpBlock(
                6,
                10,
                upsample_mode=upsample_mode,
                kernel_size=3,
                use_norm=True,
                align_corners=True,
            ),
            torch.randn(2, 6, 8, 8),
            out_channels=10,
            out_size=(size, size),
        )
        for upsample_mode, size in configurations
    ]


CASES: dict[
    type[nn.Module], Callable[[], ContractCase | list[ContractCase]]
] = {
    AttentionRefinmentBlock: lambda: feature_map(
        AttentionRefinmentBlock(8, 12),
        torch.randn(2, 8, 10, 10),
        out_channels=12,
    ),
    BlockRepeater: lambda: feature_map(
        BlockRepeater(
            ConvBlock,
            n_repeats=3,
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            padding=1,
        ),
        torch.randn(2, 4, 10, 10),
        out_channels=8,
    ),
    BottleRep: lambda: [
        feature_map(BottleRep(8, 8), torch.randn(2, 8, 10, 10)),
        feature_map(
            BottleRep(6, 10), torch.randn(2, 6, 10, 10), out_channels=10
        ),
    ],
    CSPStackRepBlock: lambda: feature_map(
        CSPStackRepBlock(8, 12, n_blocks=2),
        torch.randn(2, 8, 10, 10),
        out_channels=12,
    ),
    ConvBlock: lambda: feature_map(
        ConvBlock(6, 10, kernel_size=3, stride=2, padding=1),
        torch.randn(2, 6, 16, 12),
        out_channels=10,
        out_size=(8, 6),
    ),
    ConvStack: lambda: feature_map(
        ConvStack(4, 8, n_repeats=2), torch.randn(2, 4, 10, 10), out_channels=8
    ),
    DFL: lambda: case(
        DFL(reg_max=4),
        {"x": torch.randn(2, 16, 6, 6)},
        {"B": 2, "reg_max": 4, "H": 6, "W": 6},
    ),
    DropPath: lambda: feature_map(DropPath(0.25), torch.randn(2, 8, 10, 10)),
    EfficientDecoupledBlock: lambda: _decoupled(
        EfficientDecoupledBlock(8, n_classes=3), n_classes=3
    ),
    EncoderBlock: lambda: [
        feature_map(
            EncoderBlock(4, 8, n_repeats=2),
            torch.randn(2, 4, 16, 16),
            out_channels=8,
            out_size=(8, 8),
        ),
        feature_map(
            EncoderBlock(4, 8, n_repeats=2, max_pool=False),
            torch.randn(2, 4, 16, 16),
            out_channels=8,
        ),
    ],
    FeatureFusionBlock: _feature_fusion_block,
    GeneralReparametrizableBlock: _general_reparametrizable_block,
    GenericResidualBlock: _generic_residual_block,
    PreciseDecoupledBlock: lambda: _decoupled(
        PreciseDecoupledBlock(8, 8, 8, n_classes=3, reg_max=4),
        n_classes=3,
        reg_max=4,
    ),
    ResNetBlock: lambda: [
        feature_map(ResNetBlock(8, 8), torch.randn(2, 8, 12, 12)),
        feature_map(
            ResNetBlock(6, 8, stride=2, droppath_prob=0.1),
            torch.randn(2, 6, 12, 12),
            out_channels=8,
            out_size=(6, 6),
        ),
    ],
    ResNetBottleneck: lambda: [
        feature_map(
            ResNetBottleneck(8, 4), torch.randn(2, 8, 10, 10), out_channels=16
        ),
        feature_map(
            ResNetBottleneck(8, 4, stride=2),
            torch.randn(2, 8, 10, 10),
            out_channels=16,
            out_size=(5, 5),
        ),
    ],
    SegProto: lambda: feature_map(
        SegProto(6, mid_channels=8, out_channels=4),
        torch.randn(2, 6, 8, 8),
        out_channels=4,
        out_size=(16, 16),
    ),
    SimpleDecoder: lambda: feature_map(
        SimpleDecoder(4, 2, [1.0, 2.0]),
        torch.randn(2, 8, 4, 4),
        out_channels=2,
        out_size=(16, 16),
    ),
    SimpleDecoderBlock: lambda: feature_map(
        SimpleDecoderBlock(8, 4, **DECODER_BLOCK_KWARGS),
        torch.randn(2, 8, 6, 6),
        out_channels=4,
        out_size=(12, 12),
    ),
    SimpleEncoder: lambda: feature_map(
        SimpleEncoder(3, 4, [1.0, 2.0]),
        torch.randn(2, 3, 16, 16),
        out_channels=8,
        out_size=(4, 4),
    ),
    SpatialPyramidPoolingBlock: lambda: feature_map(
        SpatialPyramidPoolingBlock(8, 12),
        torch.randn(2, 8, 12, 12),
        out_channels=12,
    ),
    SqueezeExciteBlock: lambda: feature_map(
        SqueezeExciteBlock(8, 4), torch.randn(2, 8, 10, 10)
    ),
    UNetDecoder: _unet_decoder,
    UNetDecoderBlock: _unet_decoder_block,
    UNetEncoder: _unet_encoder,
    UpBlock: _up_block,
    UpscaleOnline: _upscale_online,
}

UNSUPPORTED: dict[type[nn.Module], str] = {}
