from typing import Literal

from pydantic import BaseModel


class RepVGGVariant(BaseModel):
    n_blocks: tuple[int, int, int, int]
    width_multiplier: tuple[float, float, float, float]


def get_variant(variant: Literal["A0", "A1", "A2"]) -> RepVGGVariant:
    variants = {
        "A0": RepVGGVariant(
            n_blocks=(2, 4, 14, 1),
            width_multiplier=(0.75, 0.75, 0.75, 2.5),
        ),
        "A1": RepVGGVariant(
            n_blocks=(2, 4, 14, 1),
            width_multiplier=(1, 1, 1, 2.5),
        ),
        "A2": RepVGGVariant(
            n_blocks=(2, 4, 14, 1),
            width_multiplier=(1.5, 1.5, 1.5, 2.75),
        ),
    }
    if variant not in variants:  # pragma: no cover
        raise ValueError(
            f"RepVGG variant should be one of "
            f"{list(variants.keys())}, got '{variant}'."
        )
    return variants[variant]
