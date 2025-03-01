from typing import Literal, TypeAlias

from pydantic import BaseModel

VariantLiteral: TypeAlias = Literal[
    "n", "nano", "s", "small", "m", "medium", "l", "large"
]


class RepPANNeckVariant(BaseModel):
    depth_multiplier: float
    width_multiplier: float
    block: Literal["RepBlock", "CSPStackRepBlock"]
    csp_e: float | None


def get_variant(variant: VariantLiteral) -> RepPANNeckVariant:
    variants = {
        "n": RepPANNeckVariant(
            depth_multiplier=0.33,
            width_multiplier=0.25,
            block="RepBlock",
            csp_e=None,
        ),
        "s": RepPANNeckVariant(
            depth_multiplier=0.33,
            width_multiplier=0.50,
            block="RepBlock",
            csp_e=None,
        ),
        "m": RepPANNeckVariant(
            depth_multiplier=0.60,
            width_multiplier=0.75,
            block="CSPStackRepBlock",
            csp_e=2 / 3,
        ),
        "l": RepPANNeckVariant(
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
    if variant in {"n", "nano"}:
        if intialize_weights:
            return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/reppanneck_n_coco.ckpt"
        return "https://github.com/luxonis/luxonis-train/releases/download/v0.1.0-beta/reppanneck_n_coco.ckpt"
    if variant in {"s", "small"}:
        if intialize_weights:
            return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/reppanneck_s_coco.ckpt"
        return None
    if variant in {"m", "medium"}:
        if intialize_weights:
            return None
        return None
    if variant in {"l", "large"}:
        if intialize_weights:
            return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/reppanneck_l_coco.ckpt"
        return None
