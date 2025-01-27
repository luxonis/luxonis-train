from typing import Literal

from pydantic import BaseModel

NET_CONFIG_rec = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 1, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [
        [5, 256, 512, 1, True],
        [5, 512, 512, 1, True],
        [5, 512, 512, 1, False],
        [5, 512, 512, 1, False],
    ],
}

NET_CONFIG_det = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [
        [5, 256, 512, 2, True],
        [5, 512, 512, 1, True],
        [5, 512, 512, 1, False],
        [5, 512, 512, 1, False],
    ],
}


class PPLCNetV3Variant(BaseModel):
    scale: float
    conv_kxk_num: int
    det: bool
    net_config: dict[str, list[list[int]]]


def get_variant(variant: Literal["rec-light"]) -> PPLCNetV3Variant:
    variants = {
        "rec-light": PPLCNetV3Variant(
            scale=0.95,
            conv_kxk_num=4,
            det=False,
            net_config=NET_CONFIG_rec,
        ),
    }
    if variant not in variants:  # pragma: no cover
        raise ValueError(
            "PPLCNetV3 model variant should be in "
            f"{list(variants.keys())}, got {variant}."
        )
    return variants[variant]
