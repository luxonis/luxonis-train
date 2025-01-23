from typing import Literal, TypeAlias

from pydantic import BaseModel

VariantLiteral: TypeAlias = Literal[
    "n", "nano", "s", "small", "m", "medium", "l", "large"
]


class EfficientVitVariant(BaseModel):
    width_list: list[int]
    depth_list: list[int]
    dim: int


def get_variant(variant: VariantLiteral) -> EfficientVitVariant:
    variants = {
        "n": EfficientVitVariant(
            width_list=[8, 16, 32, 64, 128],
            depth_list=[1, 2, 2, 2, 2],
            dim=16,
        ),
        "s": EfficientVitVariant(
            width_list=[16, 32, 64, 128, 256],
            depth_list=[1, 2, 3, 3, 4],
            dim=16,
        ),
        "m": EfficientVitVariant(
            width_list=[24, 48, 96, 192, 384],
            depth_list=[1, 3, 4, 4, 6],
            dim=32,
        ),
        "l": EfficientVitVariant(
            width_list=[32, 64, 128, 256, 512],
            depth_list=[1, 4, 6, 6, 9],
            dim=32,
        ),
    }
    variants["nano"] = variants["n"]
    variants["small"] = variants["s"]
    variants["medium"] = variants["m"]
    variants["large"] = variants["l"]

    if variant not in variants:  # pragma: no cover
        raise ValueError(
            f"EfficientVit variant should be one of "
            f"{list(variants.keys())}, got '{variant}'."
        )
    return variants[variant]
