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


def get_variant(variant: VariantLiteral) -> EfficientRepVariant:
    variants = {
        "n": EfficientRepVariant(
            depth_multiplier=0.33,
            width_multiplier=0.25,
            block="RepBlock",
            csp_e=None,
        ),
        "s": EfficientRepVariant(
            depth_multiplier=0.33,
            width_multiplier=0.50,
            block="RepBlock",
            csp_e=None,
        ),
        "m": EfficientRepVariant(
            depth_multiplier=0.60,
            width_multiplier=0.75,
            block="CSPStackRepBlock",
            csp_e=2 / 3,
        ),
        "l": EfficientRepVariant(
            depth_multiplier=1.0,
            width_multiplier=1.0,
            block="CSPStackRepBlock",
            csp_e=1 / 2,
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


def get_variant_weights(
    variant: VariantLiteral, intialize_weights: bool
) -> str | None:
    if variant == "n" or variant == "nano":
        if intialize_weights:
            return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientrep_n_coco.ckpt"
        else:
            return "https://github.com/luxonis/luxonis-train/releases/download/v0.1.0-beta/efficientrep_n_coco.ckpt"
    elif variant == "s" or variant == "small":
        if intialize_weights:
            return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientrep_s_coco.ckpt"
        else:
            return None
    elif variant == "m" or variant == "medium":
        if intialize_weights:
            return None
        else:
            return None
    elif variant == "l" or variant == "large":
        if intialize_weights:
            return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientrep_l_coco.ckpt"
        else:
            return None
