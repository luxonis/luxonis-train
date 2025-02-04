from typing import Literal

from pydantic import BaseModel
from torch import nn

from .blocks import GhostBottleneckV2


class BlockConfig(BaseModel):
    kernel_size: int
    expand_size: int
    output_channels: int
    stride: int
    se_ratio: float


class GhostFaceNetsVariant(BaseModel):
    """Variant of the GhostFaceNets embedding model.

    @type width: int
    @param width: Width multiplier. Increases complexity and number of
        parameters. Defaults to 1.0.
    @type block: nn.Module
    @param block: Ghost BottleneckV2 block. Defaults to
        GhostBottleneckV2.
    @type block_configs: list[list[BlockConfig]]
    @param block_configs: List of Ghost BottleneckV2 configurations.
    """

    width: int
    block: type[nn.Module]
    block_configs: list[list[BlockConfig]]


V2 = GhostFaceNetsVariant(
    width=1,
    block=GhostBottleneckV2,
    block_configs=[
        [
            BlockConfig(
                kernel_size=3,
                expand_size=16,
                output_channels=16,
                se_ratio=0.0,
                stride=1,
            )
        ],
        [
            BlockConfig(
                kernel_size=3,
                expand_size=48,
                output_channels=24,
                se_ratio=0.0,
                stride=2,
            )
        ],
        [
            BlockConfig(
                kernel_size=3,
                expand_size=72,
                output_channels=24,
                se_ratio=0.0,
                stride=1,
            )
        ],
        [
            BlockConfig(
                kernel_size=5,
                expand_size=72,
                output_channels=40,
                se_ratio=0.25,
                stride=2,
            )
        ],
        [
            BlockConfig(
                kernel_size=5,
                expand_size=120,
                output_channels=40,
                se_ratio=0.25,
                stride=1,
            )
        ],
        [
            BlockConfig(
                kernel_size=3,
                expand_size=240,
                output_channels=80,
                se_ratio=0.0,
                stride=2,
            )
        ],
        [
            BlockConfig(
                kernel_size=3,
                expand_size=200,
                output_channels=80,
                se_ratio=0.0,
                stride=1,
            ),
            BlockConfig(
                kernel_size=3,
                expand_size=184,
                output_channels=80,
                se_ratio=0.0,
                stride=1,
            ),
            BlockConfig(
                kernel_size=3,
                expand_size=184,
                output_channels=80,
                se_ratio=0.0,
                stride=1,
            ),
            BlockConfig(
                kernel_size=3,
                expand_size=480,
                output_channels=112,
                se_ratio=0.25,
                stride=1,
            ),
            BlockConfig(
                kernel_size=3,
                expand_size=672,
                output_channels=112,
                se_ratio=0.25,
                stride=1,
            ),
        ],
        [
            BlockConfig(
                kernel_size=5,
                expand_size=672,
                output_channels=160,
                se_ratio=0.25,
                stride=2,
            )
        ],
        [
            BlockConfig(
                kernel_size=5,
                expand_size=960,
                output_channels=160,
                se_ratio=0.0,
                stride=1,
            ),
            BlockConfig(
                kernel_size=5,
                expand_size=960,
                output_channels=160,
                se_ratio=0.25,
                stride=1,
            ),
            BlockConfig(
                kernel_size=5,
                expand_size=960,
                output_channels=160,
                se_ratio=0.0,
                stride=1,
            ),
            BlockConfig(
                kernel_size=5,
                expand_size=960,
                output_channels=160,
                se_ratio=0.25,
                stride=1,
            ),
        ],
    ],
)


def get_variant(variant: Literal["V2"]) -> GhostFaceNetsVariant:
    variants = {"V2": V2}
    if variant not in variants:  # pragma: no cover
        raise ValueError(
            "GhostFaceNets model variant should be in "
            f"{list(variants.keys())}, got {variant}."
        )
    return variants[variant].model_copy()
