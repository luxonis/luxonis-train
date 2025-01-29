from typing import List, Literal

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

    @type cfgs: List[List[BlockConfig]]
    @param cfgs: List of Ghost BottleneckV2 configurations.
    @type num_classes: int
    @param num_classes: Number of classes. Defaults to 0, which makes
        the network output the raw embeddings. Otherwise it can be used
        to add another linear layer to the network, which is useful for
        training using ArcFace or similar classification-based losses
        that require the user to drop the last layer of the network.
    @type width: int
    @param width: Width multiplier. Increases complexity and number of
        parameters. Defaults to 1.0.
    @type dropout: float
    @param dropout: Dropout rate. Defaults to 0.2.
    @type block: nn.Module
    @param block: Ghost BottleneckV2 block. Defaults to
        GhostBottleneckV2.
    @type bn_momentum: float
    @param bn_momentum: Batch normalization momentum. Defaults to 0.9.
    @type bn_epsilon: float
    @param bn_epsilon: Batch normalization epsilon. Defaults to 1e-5.
    @type init_kaiming: bool
    @param init_kaiming: If True, initializes the weights using the
        Kaiming initialization. Defaults to True.
    @type block_args: dict
    @param block_args: Arguments to pass to the block. Defaults to None.
    """

    width: int
    block: type[nn.Module]
    block_configs: List[List[BlockConfig]]


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
