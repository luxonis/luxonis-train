"""Runtime shape contract cases for the neck nodes."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import Size, nn

from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.nodes.necks.reppan_neck.blocks import (
    CSPDownBlock,
    CSPUpBlock,
    PANDownBlockBase,
    PANUpBlockBase,
    RepDownBlock,
    RepUpBlock,
)
from luxonis_train.nodes.necks.reppan_neck.reppan_neck import RepPANNeck
from luxonis_train.nodes.necks.svtr_neck.blocks import (
    Attention,
    ConvMixer,
    SVTRBlock,
)
from luxonis_train.nodes.necks.svtr_neck.svtr_neck import SVTRNeck

from .case import ContractCase

BATCH = 2

UP_IN_CHANNELS = 8
UP_SKIP_CHANNELS = 6
UP_OUT_CHANNELS = 4
UP_HEIGHT = 8
UP_WIDTH = 8

DOWN_IN_CHANNELS = 8
DOWN_MID_CHANNELS = 5
DOWN_SKIP_CHANNELS = 6
DOWN_OUT_CHANNELS = 4
DOWN_HEIGHT = 8
DOWN_WIDTH = 8

SEQUENCE_HEIGHT = 2
SEQUENCE_WIDTH = 4
SEQUENCE_LENGTH = SEQUENCE_HEIGHT * SEQUENCE_WIDTH
SEQUENCE_CHANNELS = 8

RepPANVariant = tuple[
    Literal[2, 3, 4], Literal["RepBlock", "CSPStackRepBlock"]
]
REPPAN_VARIANTS: list[RepPANVariant] = [
    (2, "RepBlock"),
    (3, "CSPStackRepBlock"),
    (4, "RepBlock"),
]

SVTRMixer = Literal["global", "local", "conv"]
SVTR_NECK_VARIANTS: list[tuple[SVTRMixer, bool, bool]] = [
    ("global", False, False),
    ("local", True, True),
    ("conv", False, False),
]
SVTR_BLOCK_VARIANTS: list[tuple[SVTRMixer, bool]] = [
    ("global", True),
    ("local", False),
    ("conv", True),
]


def _reppan_neck() -> list[ContractCase]:
    cases: list[ContractCase] = []
    for n_heads, block in REPPAN_VARIANTS:
        in_sizes = [Size((8, 32 // 2**i, 32 // 2**i)) for i in range(n_heads)]
        module = RepPANNeck(
            n_heads=n_heads,
            channels_list=[8] * 6,
            n_repeats=[2] * 4,
            depth_multiplier=1.0,
            width_multiplier=1.0,
            block=block,
            e=0.5,
            weights="none",
            in_sizes=in_sizes,
            original_in_shape=Size((3, 64, 64)),
        ).eval()
        out_channels = (4, 8, 8, 8) if n_heads == 4 else (8,) * n_heads
        cases.append(
            ContractCase(
                module=module,
                inputs={
                    "inputs": [torch.randn(BATCH, *size) for size in in_sizes]
                },
                bindings={
                    "B": BATCH,
                    "N": n_heads,
                    "C_in": tuple(size[0] for size in in_sizes),
                    "C_out": out_channels,
                    "H": tuple(size[1] for size in in_sizes),
                    "W": tuple(size[2] for size in in_sizes),
                },
            )
        )
    return cases


def _up_case(module: nn.Module) -> ContractCase:
    return ContractCase(
        module=module.eval(),
        inputs={
            "x0": torch.randn(BATCH, UP_IN_CHANNELS, UP_HEIGHT, UP_WIDTH),
            "x1": torch.randn(
                BATCH, UP_SKIP_CHANNELS, 2 * UP_HEIGHT, 2 * UP_WIDTH
            ),
        },
        bindings={
            "B": BATCH,
            "C_in": UP_IN_CHANNELS,
            "C_skip": UP_SKIP_CHANNELS,
            "C_out": UP_OUT_CHANNELS,
            "H": UP_HEIGHT,
            "W": UP_WIDTH,
        },
    )


def _down_case(module: nn.Module) -> ContractCase:
    return ContractCase(
        module=module.eval(),
        inputs={
            "x0": torch.randn(
                BATCH, DOWN_IN_CHANNELS, 2 * DOWN_HEIGHT, 2 * DOWN_WIDTH
            ),
            "x1": torch.randn(
                BATCH, DOWN_SKIP_CHANNELS, DOWN_HEIGHT, DOWN_WIDTH
            ),
        },
        bindings={
            "B": BATCH,
            "C_in": DOWN_IN_CHANNELS,
            "C_skip": DOWN_SKIP_CHANNELS,
            "C_out": DOWN_OUT_CHANNELS,
            "H": DOWN_HEIGHT,
            "W": DOWN_WIDTH,
        },
    )


def _svtr_neck() -> list[ContractCase]:
    in_channels = 16
    height = 4
    width = 4
    out_channels = 8
    cases: list[ContractCase] = []
    for mixer, use_guide, prenorm in SVTR_NECK_VARIANTS:
        module = SVTRNeck(
            dims=out_channels,
            depth=1,
            mid_channels=8,
            use_guide=use_guide,
            n_heads=2,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path=0.1,
            mixer=mixer,
            height=height,
            width=width,
            prenorm=prenorm,
            in_sizes=Size((in_channels, height, width)),
        ).eval()
        cases.append(
            ContractCase(
                module=module,
                inputs={"x": torch.randn(BATCH, in_channels, height, width)},
                bindings={
                    "B": BATCH,
                    "C_in": in_channels,
                    "C_out": out_channels,
                    "H": height,
                    "W": width,
                },
            )
        )
    return cases


def _sequence_case(module: nn.Module) -> ContractCase:
    return ContractCase(
        module=module.eval(),
        inputs={"x": torch.randn(BATCH, SEQUENCE_LENGTH, SEQUENCE_CHANNELS)},
        bindings={
            "B": BATCH,
            "L": SEQUENCE_LENGTH,
            "C": SEQUENCE_CHANNELS,
        },
    )


def _attention() -> list[ContractCase]:
    return [
        _sequence_case(
            Attention(SEQUENCE_CHANNELS, n_heads=2, mixer="global")
        ),
        _sequence_case(
            Attention(
                SEQUENCE_CHANNELS,
                n_heads=2,
                mixer="local",
                height=SEQUENCE_HEIGHT,
                width=SEQUENCE_WIDTH,
                kernel_size=(3, 3),
            )
        ),
    ]


def _svtr_block() -> list[ContractCase]:
    return [
        _sequence_case(
            SVTRBlock(
                SEQUENCE_CHANNELS,
                n_heads=2,
                mixer=mixer,
                height=SEQUENCE_HEIGHT,
                width=SEQUENCE_WIDTH,
                mixer_kernel_size=(3, 3),
                drop_path=0.1,
                prenorm=prenorm,
            )
        )
        for mixer, prenorm in SVTR_BLOCK_VARIANTS
    ]


CASES: dict[
    type[nn.Module], Callable[[], ContractCase | list[ContractCase]]
] = {
    RepPANNeck: _reppan_neck,
    PANUpBlockBase: lambda: _up_case(
        PANUpBlockBase(
            in_channels=UP_IN_CHANNELS,
            out_channels=UP_OUT_CHANNELS,
            encode_block=ConvBlock(
                in_channels=UP_SKIP_CHANNELS + UP_OUT_CHANNELS,
                out_channels=UP_OUT_CHANNELS,
                kernel_size=1,
            ),
        )
    ),
    RepUpBlock: lambda: _up_case(
        RepUpBlock(
            in_channels=UP_IN_CHANNELS,
            in_channels_next=UP_SKIP_CHANNELS,
            out_channels=UP_OUT_CHANNELS,
            n_repeats=2,
        )
    ),
    CSPUpBlock: lambda: _up_case(
        CSPUpBlock(
            in_channels=UP_IN_CHANNELS,
            in_channels_next=UP_SKIP_CHANNELS,
            out_channels=UP_OUT_CHANNELS,
            n_repeats=2,
            e=0.5,
        )
    ),
    PANDownBlockBase: lambda: _down_case(
        PANDownBlockBase(
            in_channels=DOWN_IN_CHANNELS,
            downsample_out_channels=DOWN_MID_CHANNELS,
            encode_block=ConvBlock(
                in_channels=DOWN_MID_CHANNELS + DOWN_SKIP_CHANNELS,
                out_channels=DOWN_OUT_CHANNELS,
                kernel_size=1,
            ),
        )
    ),
    RepDownBlock: lambda: _down_case(
        RepDownBlock(
            in_channels=DOWN_IN_CHANNELS,
            downsample_out_channels=DOWN_MID_CHANNELS,
            in_channels_next=DOWN_SKIP_CHANNELS,
            out_channels=DOWN_OUT_CHANNELS,
            n_repeats=2,
        )
    ),
    CSPDownBlock: lambda: _down_case(
        CSPDownBlock(
            in_channels=DOWN_IN_CHANNELS,
            downsample_out_channels=DOWN_MID_CHANNELS,
            in_channels_next=DOWN_SKIP_CHANNELS,
            out_channels=DOWN_OUT_CHANNELS,
            n_repeats=2,
            e=0.5,
        )
    ),
    SVTRNeck: _svtr_neck,
    ConvMixer: lambda: _sequence_case(
        ConvMixer(
            SEQUENCE_CHANNELS,
            height=SEQUENCE_HEIGHT,
            width=SEQUENCE_WIDTH,
            n_heads=2,
        )
    ),
    Attention: _attention,
    SVTRBlock: _svtr_block,
}

UNSUPPORTED: dict[type[nn.Module], str] = {}
