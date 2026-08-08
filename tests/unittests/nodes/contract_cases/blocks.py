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

from .case import ContractCase

UpsampleMode = Literal["simple_upsample", "conv_upsample", "conv_transpose"]


def _conv_block() -> ContractCase:
    module = ConvBlock(6, 10, kernel_size=3, stride=2, padding=1).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 6, 16, 12)},
        bindings={
            "B": 2,
            "C_in": 6,
            "C_out": 10,
            "H": 16,
            "W": 12,
            "H_out": 8,
            "W_out": 6,
        },
    )


def _squeeze_excite_block() -> ContractCase:
    module = SqueezeExciteBlock(8, 4).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 8, 10, 10)},
        bindings={"B": 2, "C_in": 8, "H": 10, "W": 10},
    )


def _drop_path() -> ContractCase:
    module = DropPath(0.25).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 8, 10, 10)},
        bindings={"B": 2, "C_in": 8, "H": 10, "W": 10},
    )


def _precise_decoupled_block() -> ContractCase:
    module = PreciseDecoupledBlock(8, 8, 8, n_classes=3, reg_max=4).eval()
    x = torch.randn(2, 8, 10, 10)
    features, classes, regressions = module(x)
    return ContractCase(
        module=module,
        inputs={"x": x},
        bindings={
            "B": 2,
            "C_in": 8,
            "H": 10,
            "W": 10,
            "n_classes": 3,
            "reg_max": 4,
        },
        outputs={
            "features": features,
            "classes": classes,
            "regressions": regressions,
        },
    )


def _efficient_decoupled_block() -> ContractCase:
    module = EfficientDecoupledBlock(8, n_classes=3).eval()
    x = torch.randn(2, 8, 10, 10)
    features, classes, regressions = module(x)
    return ContractCase(
        module=module,
        inputs={"x": x},
        bindings={
            "B": 2,
            "C_in": 8,
            "H": 10,
            "W": 10,
            "n_classes": 3,
        },
        outputs={
            "features": features,
            "classes": classes,
            "regressions": regressions,
        },
    )


def _seg_proto() -> ContractCase:
    module = SegProto(6, mid_channels=8, out_channels=4).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 6, 8, 8)},
        bindings={"B": 2, "C_in": 6, "C_out": 4, "H": 8, "W": 8},
    )


def _dfl() -> ContractCase:
    module = DFL(reg_max=4).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 16, 6, 6)},
        bindings={"B": 2, "reg_max": 4, "H": 6, "W": 6},
    )


def _general_reparametrizable_block() -> list[ContractCase]:
    residual = GeneralReparametrizableBlock(8, 8, n_branches=2).eval()
    fused = GeneralReparametrizableBlock(8, 8, n_branches=2).eval()
    fused.reparametrize()
    strided = GeneralReparametrizableBlock(
        8, 16, stride=2, refine_block="se"
    ).eval()
    preserving_bindings = {
        "B": 2,
        "C_in": 8,
        "C_out": 8,
        "H": 12,
        "W": 12,
        "H_out": 12,
        "W_out": 12,
    }
    return [
        ContractCase(
            module=residual,
            inputs={"x": torch.randn(2, 8, 12, 12)},
            bindings=dict(preserving_bindings),
        ),
        ContractCase(
            module=fused,
            inputs={"x": torch.randn(2, 8, 12, 12)},
            bindings=dict(preserving_bindings),
        ),
        ContractCase(
            module=strided,
            inputs={"x": torch.randn(2, 8, 12, 12)},
            bindings={
                "B": 2,
                "C_in": 8,
                "C_out": 16,
                "H": 12,
                "W": 12,
                "H_out": 6,
                "W_out": 6,
            },
        ),
    ]


def _block_repeater() -> ContractCase:
    module = BlockRepeater(
        ConvBlock,
        n_repeats=3,
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        padding=1,
    ).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 4, 10, 10)},
        bindings={
            "B": 2,
            "C_in": 4,
            "C_out": 8,
            "H": 10,
            "W": 10,
            "H_out": 10,
            "W_out": 10,
        },
    )


def _conv_stack() -> ContractCase:
    module = ConvStack(4, 8, n_repeats=2).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 4, 10, 10)},
        bindings={
            "B": 2,
            "C_in": 4,
            "C_out": 8,
            "H": 10,
            "W": 10,
            "H_out": 10,
            "W_out": 10,
        },
    )


def _csp_stack_rep_block() -> ContractCase:
    module = CSPStackRepBlock(8, 12, n_blocks=2).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 8, 10, 10)},
        bindings={
            "B": 2,
            "C_in": 8,
            "C_out": 12,
            "H": 10,
            "W": 10,
            "H_out": 10,
            "W_out": 10,
        },
    )


def _bottle_rep() -> list[ContractCase]:
    shortcut = BottleRep(8, 8).eval()
    projecting = BottleRep(6, 10).eval()
    return [
        ContractCase(
            module=shortcut,
            inputs={"x": torch.randn(2, 8, 10, 10)},
            bindings={
                "B": 2,
                "C_in": 8,
                "C_out": 8,
                "H": 10,
                "W": 10,
                "H_out": 10,
                "W_out": 10,
            },
        ),
        ContractCase(
            module=projecting,
            inputs={"x": torch.randn(2, 6, 10, 10)},
            bindings={
                "B": 2,
                "C_in": 6,
                "C_out": 10,
                "H": 10,
                "W": 10,
                "H_out": 10,
                "W_out": 10,
            },
        ),
    ]


def _spatial_pyramid_pooling_block() -> ContractCase:
    module = SpatialPyramidPoolingBlock(8, 12).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 8, 12, 12)},
        bindings={
            "B": 2,
            "C_in": 8,
            "C_out": 12,
            "H": 12,
            "W": 12,
            "H_out": 12,
            "W_out": 12,
        },
    )


def _attention_refinment_block() -> ContractCase:
    module = AttentionRefinmentBlock(8, 12).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 8, 10, 10)},
        bindings={"B": 2, "C_in": 8, "C_out": 12, "H": 10, "W": 10},
    )


def _feature_fusion_block() -> ContractCase:
    module = FeatureFusionBlock(12, 8).eval()
    return ContractCase(
        module=module,
        inputs={
            "x1": torch.randn(2, 4, 10, 10),
            "x2": torch.randn(2, 8, 10, 10),
        },
        bindings={"B": 2, "C1": 4, "C2": 8, "C_out": 8, "H": 10, "W": 10},
    )


def _upscale_online() -> ContractCase:
    module = UpscaleOnline().eval()
    return ContractCase(
        module=module,
        inputs={
            "x": torch.randn(2, 6, 8, 8),
            "output_height": 13,
            "output_width": 11,
        },
        bindings={
            "B": 2,
            "C_in": 6,
            "H": 8,
            "W": 8,
            "output_height": 13,
            "output_width": 11,
        },
    )


def _generic_residual_block() -> list[ContractCase]:
    identity_shortcut = GenericResidualBlock(
        in_channels=8,
        hidden_channels=4,
        stride=1,
        expansion=2,
        final_relu=True,
        block=ConvBlock(8, 8, kernel_size=3, padding=1),
    ).eval()
    projected_shortcut = GenericResidualBlock(
        in_channels=6,
        hidden_channels=4,
        stride=2,
        expansion=2,
        final_relu=False,
        block=ConvBlock(6, 8, kernel_size=3, stride=2, padding=1),
    ).eval()
    return [
        ContractCase(
            module=identity_shortcut,
            inputs={"x": torch.randn(2, 8, 12, 12)},
            bindings={
                "B": 2,
                "C_in": 8,
                "C_out": 8,
                "H": 12,
                "W": 12,
                "H_out": 12,
                "W_out": 12,
            },
        ),
        ContractCase(
            module=projected_shortcut,
            inputs={"x": torch.randn(2, 6, 12, 12)},
            bindings={
                "B": 2,
                "C_in": 6,
                "C_out": 8,
                "H": 12,
                "W": 12,
                "H_out": 6,
                "W_out": 6,
            },
        ),
    ]


def _resnet_block() -> list[ContractCase]:
    return [
        ContractCase(
            module=ResNetBlock(8, 8).eval(),
            inputs={"x": torch.randn(2, 8, 12, 12)},
            bindings={
                "B": 2,
                "C_in": 8,
                "C_out": 8,
                "H": 12,
                "W": 12,
                "H_out": 12,
                "W_out": 12,
            },
        ),
        ContractCase(
            module=ResNetBlock(6, 8, stride=2, droppath_prob=0.1).eval(),
            inputs={"x": torch.randn(2, 6, 12, 12)},
            bindings={
                "B": 2,
                "C_in": 6,
                "C_out": 8,
                "H": 12,
                "W": 12,
                "H_out": 6,
                "W_out": 6,
            },
        ),
    ]


def _resnet_bottleneck() -> list[ContractCase]:
    return [
        ContractCase(
            module=ResNetBottleneck(8, 4).eval(),
            inputs={"x": torch.randn(2, 8, 10, 10)},
            bindings={
                "B": 2,
                "C_in": 8,
                "C_out": 16,
                "H": 10,
                "W": 10,
                "H_out": 10,
                "W_out": 10,
            },
        ),
        ContractCase(
            module=ResNetBottleneck(8, 4, stride=2).eval(),
            inputs={"x": torch.randn(2, 8, 10, 10)},
            bindings={
                "B": 2,
                "C_in": 8,
                "C_out": 16,
                "H": 10,
                "W": 10,
                "H_out": 5,
                "W_out": 5,
            },
        ),
    ]


def _encoder_block() -> list[ContractCase]:
    return [
        ContractCase(
            module=EncoderBlock(4, 8, n_repeats=2).eval(),
            inputs={"x": torch.randn(2, 4, 16, 16)},
            bindings={
                "B": 2,
                "C_in": 4,
                "C_out": 8,
                "H": 16,
                "W": 16,
                "H_out": 8,
                "W_out": 8,
            },
        ),
        ContractCase(
            module=EncoderBlock(4, 8, n_repeats=2, max_pool=False).eval(),
            inputs={"x": torch.randn(2, 4, 16, 16)},
            bindings={
                "B": 2,
                "C_in": 4,
                "C_out": 8,
                "H": 16,
                "W": 16,
                "H_out": 16,
                "W_out": 16,
            },
        ),
    ]


def _simple_encoder() -> ContractCase:
    module = SimpleEncoder(3, 4, [1.0, 2.0]).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 3, 16, 16)},
        bindings={
            "B": 2,
            "C_in": 3,
            "C_out": 8,
            "H": 16,
            "W": 16,
            "H_out": 4,
            "W_out": 4,
        },
    )


def _unet_encoder() -> ContractCase:
    module = UNetEncoder(3, 4, [1.0, 2.0]).eval()
    x = torch.randn(2, 3, 16, 16)
    return ContractCase(
        module=module,
        inputs={"x": x},
        bindings={
            "B": 2,
            "C_in": 3,
            "H_in": 16,
            "W_in": 16,
            "N": 3,
            "C": (4, 8, 8),
            "H": (16, 8, 4),
            "W": (16, 8, 4),
        },
        outputs={"features": module(x)},
    )


def _simple_decoder_block() -> ContractCase:
    module = SimpleDecoderBlock(
        8,
        4,
        kernel_size=3,
        use_norm=True,
        align_corners=True,
        upsample_mode="simple_upsample",
        n_repeats=2,
    ).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 8, 6, 6)},
        bindings={
            "B": 2,
            "C_in": 8,
            "C_out": 4,
            "H": 6,
            "W": 6,
            "H_out": 12,
            "W_out": 12,
        },
    )


def _unet_decoder_block() -> ContractCase:
    module = UNetDecoderBlock(
        8,
        4,
        kernel_size=3,
        use_norm=True,
        align_corners=True,
        upsample_mode="simple_upsample",
        n_repeats=2,
    ).eval()
    return ContractCase(
        module=module,
        inputs={
            "x": torch.randn(2, 8, 6, 6),
            "skip_x": torch.randn(2, 8, 12, 12),
        },
        bindings={
            "B": 2,
            "C_in": 8,
            "C_skip": 8,
            "C_out": 4,
            "H": 6,
            "W": 6,
        },
    )


def _simple_decoder() -> ContractCase:
    module = SimpleDecoder(4, 2, [1.0, 2.0]).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(2, 8, 4, 4)},
        bindings={
            "B": 2,
            "C_in": 8,
            "C_out": 2,
            "H": 4,
            "W": 4,
            "H_out": 16,
            "W_out": 16,
        },
    )


def _unet_decoder() -> ContractCase:
    encoder = UNetEncoder(3, 4, [1.0, 2.0]).eval()
    module = UNetDecoder(4, 2, [1.0, 2.0]).eval()
    features = encoder(torch.randn(2, 3, 16, 16))
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
    cases = []
    configurations: list[tuple[UpsampleMode, int]] = [
        ("simple_upsample", 16),
        ("conv_upsample", 16),
        ("conv_transpose", 17),
    ]
    for upsample_mode, height_out in configurations:
        module = UpBlock(
            6,
            10,
            upsample_mode=upsample_mode,
            kernel_size=3,
            use_norm=True,
            align_corners=True,
        ).eval()
        cases.append(
            ContractCase(
                module=module,
                inputs={"x": torch.randn(2, 6, 8, 8)},
                bindings={
                    "B": 2,
                    "C_in": 6,
                    "C_out": 10,
                    "H": 8,
                    "W": 8,
                    "H_out": height_out,
                    "W_out": height_out,
                },
            )
        )
    return cases


CASES: dict[
    type[nn.Module], Callable[[], ContractCase | list[ContractCase]]
] = {
    AttentionRefinmentBlock: _attention_refinment_block,
    BlockRepeater: _block_repeater,
    BottleRep: _bottle_rep,
    CSPStackRepBlock: _csp_stack_rep_block,
    ConvBlock: _conv_block,
    ConvStack: _conv_stack,
    DFL: _dfl,
    DropPath: _drop_path,
    EfficientDecoupledBlock: _efficient_decoupled_block,
    EncoderBlock: _encoder_block,
    FeatureFusionBlock: _feature_fusion_block,
    GeneralReparametrizableBlock: _general_reparametrizable_block,
    GenericResidualBlock: _generic_residual_block,
    PreciseDecoupledBlock: _precise_decoupled_block,
    ResNetBlock: _resnet_block,
    ResNetBottleneck: _resnet_bottleneck,
    SegProto: _seg_proto,
    SimpleDecoder: _simple_decoder,
    SimpleDecoderBlock: _simple_decoder_block,
    SimpleEncoder: _simple_encoder,
    SpatialPyramidPoolingBlock: _spatial_pyramid_pooling_block,
    SqueezeExciteBlock: _squeeze_excite_block,
    UNetDecoder: _unet_decoder,
    UNetDecoderBlock: _unet_decoder_block,
    UNetEncoder: _unet_encoder,
    UpBlock: _up_block,
    UpscaleOnline: _upscale_online,
}

UNSUPPORTED: dict[type[nn.Module], str] = {}
