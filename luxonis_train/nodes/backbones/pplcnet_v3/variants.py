from typing import Literal

from pydantic import BaseModel


class BlockConfig(BaseModel):
    kernel_size: int
    in_channels: int
    out_channels: int
    stride: int
    use_se: bool


class PPLCNetV3Variant(BaseModel):
    scale: float
    n_branches: int
    use_detection_backbone: bool
    net_config: list[list[BlockConfig]]


def get_variant(variant: Literal["rec-light"]) -> PPLCNetV3Variant:
    variants = {
        "rec-light": PPLCNetV3Variant(
            scale=0.95,
            n_branches=4,
            use_detection_backbone=False,
            net_config=[
                [
                    BlockConfig(
                        kernel_size=3,
                        in_channels=16,
                        out_channels=32,
                        stride=1,
                        use_se=False,
                    )
                ],
                [
                    BlockConfig(
                        kernel_size=3,
                        in_channels=32,
                        out_channels=64,
                        stride=2,
                        use_se=False,
                    ),
                    BlockConfig(
                        kernel_size=3,
                        in_channels=64,
                        out_channels=64,
                        stride=1,
                        use_se=False,
                    ),
                ],
                [
                    BlockConfig(
                        kernel_size=3,
                        in_channels=64,
                        out_channels=128,
                        stride=1,
                        use_se=False,
                    ),
                    BlockConfig(
                        kernel_size=3,
                        in_channels=128,
                        out_channels=128,
                        stride=1,
                        use_se=False,
                    ),
                ],
                [
                    BlockConfig(
                        kernel_size=3,
                        in_channels=128,
                        out_channels=256,
                        stride=2,
                        use_se=False,
                    ),
                    BlockConfig(
                        kernel_size=5,
                        in_channels=256,
                        out_channels=256,
                        stride=1,
                        use_se=False,
                    ),
                    BlockConfig(
                        kernel_size=5,
                        in_channels=256,
                        out_channels=256,
                        stride=1,
                        use_se=False,
                    ),
                    BlockConfig(
                        kernel_size=5,
                        in_channels=256,
                        out_channels=256,
                        stride=1,
                        use_se=False,
                    ),
                    BlockConfig(
                        kernel_size=5,
                        in_channels=256,
                        out_channels=256,
                        stride=1,
                        use_se=False,
                    ),
                ],
                [
                    BlockConfig(
                        kernel_size=5,
                        in_channels=256,
                        out_channels=512,
                        stride=1,
                        use_se=True,
                    ),
                    BlockConfig(
                        kernel_size=5,
                        in_channels=512,
                        out_channels=512,
                        stride=1,
                        use_se=True,
                    ),
                    BlockConfig(
                        kernel_size=5,
                        in_channels=512,
                        out_channels=512,
                        stride=1,
                        use_se=False,
                    ),
                    BlockConfig(
                        kernel_size=5,
                        in_channels=512,
                        out_channels=512,
                        stride=1,
                        use_se=False,
                    ),
                ],
            ],
        )
    }
    if variant not in variants:  # pragma: no cover
        raise ValueError(
            "PPLCNetV3 model variant should be in "
            f"{list(variants.keys())}, got {variant}."
        )
    return variants[variant]
