from typing import Literal

from pydantic import BaseModel


class MobileOneVariant(BaseModel):
    width_multipliers: tuple[float, float, float, float]
    n_conv_branches: int = 1
    use_se: bool = False


def get_variant(
    variant: Literal["s0", "s1", "s2", "s3", "s4"],
) -> MobileOneVariant:
    variants = {
        "s0": MobileOneVariant(
            width_multipliers=(0.75, 1.0, 1.0, 2.0),
            n_conv_branches=4,
        ),
        "s1": MobileOneVariant(
            width_multipliers=(1.5, 1.5, 2.0, 2.5),
        ),
        "s2": MobileOneVariant(
            width_multipliers=(1.5, 2.0, 2.5, 4.0),
        ),
        "s3": MobileOneVariant(
            width_multipliers=(2.0, 2.5, 3.0, 4.0),
        ),
        "s4": MobileOneVariant(
            width_multipliers=(3.0, 3.5, 3.5, 4.0),
            use_se=True,
        ),
    }
    if variant not in variants:  # pragma: no cover
        raise ValueError(
            "MobileOne model variant should be in "
            f"{list(variants.keys())}, got {variant}."
        )
    return variants[variant]
