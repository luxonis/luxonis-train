from typing import Literal, TypeAlias

from pydantic import BaseModel

VariantLiteral: TypeAlias = Literal[
    "n", "nano", "s", "small", "m", "medium", "l", "large"
]


class EfficientRepVariant(BaseModel):
    depth_multiplier: float
    width_multiplier: float
    block: Literal["RepBlock", "CSPStackRepBlock"]
    csp_e: float | None
    weights_path: str | None


def get_variant(variant: VariantLiteral) -> EfficientRepVariant:
    variants = {
        "n": EfficientRepVariant(
            depth_multiplier=0.33,
            width_multiplier=0.25,
            block="RepBlock",
            csp_e=None,
            weights_path="https://github.com/luxonis/luxonis-train/releases/download/v0.1.0-beta/efficientrep_n_coco.ckpt",
        ),
        "s": EfficientRepVariant(
            depth_multiplier=0.33,
            width_multiplier=0.50,
            block="RepBlock",
            csp_e=None,
            weights_path=None,
        ),
        "m": EfficientRepVariant(
            depth_multiplier=0.60,
            width_multiplier=0.75,
            block="CSPStackRepBlock",
            csp_e=2 / 3,
            weights_path=None,
        ),
        "l": EfficientRepVariant(
            depth_multiplier=1.0,
            width_multiplier=1.0,
            block="CSPStackRepBlock",
            csp_e=1 / 2,
            weights_path=None,
        ),
    }
    variants["nano"] = variants["n"]
    variants["small"] = variants["s"]
    variants["medium"] = variants["m"]
    variants["large"] = variants["l"]

    if variant not in variants:  # pragma: no cover
        raise ValueError(
            f"EfficientRep variant should be one of "
            f"{list(variants.keys())}, got '{variant}'."
        )
    return variants[variant]
