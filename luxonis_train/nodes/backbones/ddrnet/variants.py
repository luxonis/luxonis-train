from typing import Literal

from pydantic import BaseModel


class DDRNetVariant(BaseModel):
    channels: int = 32
    highres_channels: int = 64
    weights_path: str | None


def get_variant(variant: Literal["23-slim", "23"]) -> DDRNetVariant:
    variants = {
        "23-slim": DDRNetVariant(
            channels=32,
            highres_channels=64,
            weights_path="https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/ddrnet_23slim_coco.ckpt",
        ),
        "23": DDRNetVariant(
            channels=64,
            highres_channels=128,
            weights_path="https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/ddrnet_23_coco.ckpt",
        ),
    }
    if variant not in variants:  # pragma: no cover
        raise ValueError(
            "DDRNet model variant should be in "
            f"{list(variants.keys())}, got {variant}."
        )
    return variants[variant]
