from typing import Literal

from pydantic import BaseModel


class MicroBlockConfig(BaseModel):
    stride: int
    out_channels: int
    kernel_size: int
    expand_ratio: tuple[int, int]
    groups_1: tuple[int, int]
    groups_2: tuple[int, int]
    dy_shifts: tuple[int, int, int]
    reduction_factor: int


class MicroNetVariant(BaseModel):
    stem_channels: int
    stem_groups: tuple[int, int]
    init_a: tuple[float, float]
    init_b: tuple[float, float]
    out_indices: list[int]
    block_configs: list[MicroBlockConfig]


M1 = MicroNetVariant(
    stem_channels=6,
    stem_groups=(3, 2),
    init_a=(1.0, 1.0),
    init_b=(0.0, 0.0),
    out_indices=[1, 2, 4, 7],
    block_configs=[
        MicroBlockConfig(
            stride=2,
            out_channels=8,
            kernel_size=3,
            expand_ratio=(2, 2),
            groups_1=(0, 6),
            groups_2=(2, 2),
            dy_shifts=(2, 0, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=16,
            kernel_size=3,
            expand_ratio=(2, 2),
            groups_1=(0, 8),
            groups_2=(4, 4),
            dy_shifts=(2, 2, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=16,
            kernel_size=5,
            expand_ratio=(2, 2),
            groups_1=(0, 16),
            groups_2=(4, 4),
            dy_shifts=(2, 2, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=32,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(4, 4),
            groups_2=(4, 4),
            dy_shifts=(2, 2, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=64,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(8, 8),
            groups_2=(8, 8),
            dy_shifts=(2, 2, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=96,
            kernel_size=3,
            expand_ratio=(1, 6),
            groups_1=(8, 8),
            groups_2=(8, 8),
            dy_shifts=(2, 2, 1),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=576,
            kernel_size=3,
            expand_ratio=(1, 6),
            groups_1=(12, 12),
            groups_2=(0, 0),
            dy_shifts=(2, 2, 1),
            reduction_factor=2,
        ),
    ],
)

M2 = MicroNetVariant(
    stem_channels=8,
    stem_groups=(4, 2),
    init_a=(1.0, 1.0),
    init_b=(0.0, 0.0),
    out_indices=[1, 3, 6, 9],
    block_configs=[
        MicroBlockConfig(
            stride=2,
            out_channels=12,
            kernel_size=3,
            expand_ratio=(2, 2),
            groups_1=(0, 8),
            groups_2=(4, 4),
            dy_shifts=(2, 0, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=16,
            kernel_size=3,
            expand_ratio=(2, 2),
            groups_1=(0, 12),
            groups_2=(4, 4),
            dy_shifts=(2, 2, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=24,
            kernel_size=3,
            expand_ratio=(2, 2),
            groups_1=(0, 16),
            groups_2=(4, 4),
            dy_shifts=(2, 2, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=32,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(6, 6),
            groups_2=(4, 4),
            dy_shifts=(2, 2, 1),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=32,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(8, 8),
            groups_2=(4, 4),
            dy_shifts=(2, 2, 1),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=64,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(8, 8),
            groups_2=(8, 8),
            dy_shifts=(2, 2, 1),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=96,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(8, 8),
            groups_2=(8, 8),
            dy_shifts=(2, 2, 1),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=128,
            kernel_size=3,
            expand_ratio=(1, 6),
            groups_1=(12, 12),
            groups_2=(8, 8),
            dy_shifts=(2, 2, 1),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=768,
            kernel_size=3,
            expand_ratio=(1, 6),
            groups_1=(16, 16),
            groups_2=(0, 0),
            dy_shifts=(2, 2, 1),
            reduction_factor=2,
        ),
    ],
)

M3 = MicroNetVariant(
    stem_channels=12,
    stem_groups=(4, 3),
    init_a=(1.0, 0.5),
    init_b=(0.0, 0.5),
    out_indices=[1, 3, 8, 12],
    block_configs=[
        MicroBlockConfig(
            stride=2,
            out_channels=16,
            kernel_size=3,
            expand_ratio=(2, 2),
            groups_1=(0, 12),
            groups_2=(4, 4),
            dy_shifts=(0, 2, 0),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=24,
            kernel_size=3,
            expand_ratio=(2, 2),
            groups_1=(0, 16),
            groups_2=(4, 4),
            dy_shifts=(0, 2, 0),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=24,
            kernel_size=3,
            expand_ratio=(2, 2),
            groups_1=(0, 24),
            groups_2=(4, 4),
            dy_shifts=(0, 2, 0),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=32,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(6, 6),
            groups_2=(4, 4),
            dy_shifts=(0, 2, 0),
            reduction_factor=1,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=32,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(8, 8),
            groups_2=(4, 4),
            dy_shifts=(0, 2, 0),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=64,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(8, 8),
            groups_2=(8, 8),
            dy_shifts=(0, 2, 0),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=2,
            out_channels=80,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(8, 8),
            groups_2=(8, 8),
            dy_shifts=(0, 2, 0),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=80,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(10, 10),
            groups_2=(8, 8),
            dy_shifts=(0, 2, 0),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=120,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(10, 10),
            groups_2=(10, 10),
            dy_shifts=(0, 2, 0),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=120,
            kernel_size=5,
            expand_ratio=(1, 6),
            groups_1=(12, 12),
            groups_2=(10, 10),
            dy_shifts=(0, 2, 0),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=144,
            kernel_size=3,
            expand_ratio=(1, 6),
            groups_1=(12, 12),
            groups_2=(12, 12),
            dy_shifts=(0, 2, 0),
            reduction_factor=2,
        ),
        MicroBlockConfig(
            stride=1,
            out_channels=864,
            kernel_size=3,
            expand_ratio=(1, 6),
            groups_1=(12, 12),
            groups_2=(0, 0),
            dy_shifts=(0, 2, 0),
            reduction_factor=2,
        ),
    ],
)


def get_variant(variant: Literal["M1", "M2", "M3"]) -> MicroNetVariant:
    variants = {"M1": M1, "M2": M2, "M3": M3}
    if variant not in variants:  # pragma: no cover
        raise ValueError(
            "MicroNet model variant should be in "
            f"{list(variants.keys())}, got {variant}."
        )
    return variants[variant]
