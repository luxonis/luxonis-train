from typing import Any, Literal

from pydantic import BaseModel

# The type of "large" or "small" config is a list. Each element(list) represents a depthwise block, which is composed of k, exp, se, act, s.
# k: kernel_size
# exp: middle channel number in depthwise block
# c: output channel number in depthwise block
# se: whether to use SE block
# act: which activation to use
# s: stride in depthwise block
# d: dilation rate in depthwise block
NET_CONFIG = {
    "large": [
        # k, exp, c, se, act, s
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],  # x4
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],  # x8
        [3, 240, 80, False, "hardswish", 2],
        [3, 200, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 480, 112, True, "hardswish", 1],
        [3, 672, 112, True, "hardswish", 1],  # x16
        [5, 672, 160, True, "hardswish", 2],
        [5, 960, 160, True, "hardswish", 1],
        [5, 960, 160, True, "hardswish", 1],  # x32
    ],
    "small": [
        # k, exp, c, se, act, s
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hardswish", 2],
        [5, 240, 40, True, "hardswish", 1],
        [5, 240, 40, True, "hardswish", 1],
        [5, 120, 48, True, "hardswish", 1],
        [5, 144, 48, True, "hardswish", 1],
        [5, 288, 96, True, "hardswish", 2],
        [5, 576, 96, True, "hardswish", 1],
        [5, 576, 96, True, "hardswish", 1],
    ],
    "large_os8": [
        # k, exp, c, se, act, s, {d}
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],  # x4
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],  # x8
        [3, 240, 80, False, "hardswish", 1],
        [3, 200, 80, False, "hardswish", 1, 2],
        [3, 184, 80, False, "hardswish", 1, 2],
        [3, 184, 80, False, "hardswish", 1, 2],
        [3, 480, 112, True, "hardswish", 1, 2],
        [3, 672, 112, True, "hardswish", 1, 2],
        [5, 672, 160, True, "hardswish", 1, 2],
        [5, 960, 160, True, "hardswish", 1, 4],
        [5, 960, 160, True, "hardswish", 1, 4],
    ],
    "small_os8": [
        # k, exp, c, se, act, s, {d}
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hardswish", 1],
        [5, 240, 40, True, "hardswish", 1, 2],
        [5, 240, 40, True, "hardswish", 1, 2],
        [5, 120, 48, True, "hardswish", 1, 2],
        [5, 144, 48, True, "hardswish", 1, 2],
        [5, 288, 96, True, "hardswish", 1, 2],
        [5, 576, 96, True, "hardswish", 1, 4],
        [5, 576, 96, True, "hardswish", 1, 4],
    ],
}


class MobileNetV3Variant(BaseModel):
    net_config: list[list[Any]]
    scale: float
    stages_pattern: list[str]
    out_index: list[int]


SMALL_X0_35 = MobileNetV3Variant(
    net_config=NET_CONFIG["small"],
    scale=0.35,
    stages_pattern=["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    out_index=[0, 2, 7, 10],
)

SMALL_X0_5 = MobileNetV3Variant(
    net_config=NET_CONFIG["small"],
    scale=0.5,
    stages_pattern=["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    out_index=[0, 2, 7, 10],
)

SMALL_X0_75 = MobileNetV3Variant(
    net_config=NET_CONFIG["small"],
    scale=0.75,
    stages_pattern=["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    out_index=[0, 2, 7, 10],
)

SMALL_X1_0 = MobileNetV3Variant(
    net_config=NET_CONFIG["small"],
    scale=1.0,
    stages_pattern=["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    out_index=[0, 2, 7, 10],
)

SMALL_X1_25 = MobileNetV3Variant(
    net_config=NET_CONFIG["small"],
    scale=1.25,
    stages_pattern=["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    out_index=[0, 2, 7, 10],
)

LARGE_X0_35 = MobileNetV3Variant(
    net_config=NET_CONFIG["large"],
    scale=0.35,
    stages_pattern=[
        "blocks[0]",
        "blocks[2]",
        "blocks[5]",
        "blocks[11]",
        "blocks[14]",
    ],
    out_index=[2, 5, 11, 14],
)

LARGE_X0_5 = MobileNetV3Variant(
    net_config=NET_CONFIG["large"],
    scale=0.5,
    stages_pattern=[
        "blocks[0]",
        "blocks[2]",
        "blocks[5]",
        "blocks[11]",
        "blocks[14]",
    ],
    out_index=[2, 5, 11, 14],
)

LARGE_X0_75 = MobileNetV3Variant(
    net_config=NET_CONFIG["large"],
    scale=0.75,
    stages_pattern=[
        "blocks[0]",
        "blocks[2]",
        "blocks[5]",
        "blocks[11]",
        "blocks[14]",
    ],
    out_index=[2, 5, 11, 14],
)

LARGE_X1_0 = MobileNetV3Variant(
    net_config=NET_CONFIG["large"],
    scale=1.0,
    stages_pattern=[
        "blocks[0]",
        "blocks[2]",
        "blocks[5]",
        "blocks[11]",
        "blocks[14]",
    ],
    out_index=[2, 5, 11, 14],
)

LARGE_X1_25 = MobileNetV3Variant(
    net_config=NET_CONFIG["large"],
    scale=1.25,
    stages_pattern=[
        "blocks[0]",
        "blocks[2]",
        "blocks[5]",
        "blocks[11]",
        "blocks[14]",
    ],
    out_index=[2, 5, 11, 14],
)

SMALL_X1_0_OS8 = MobileNetV3Variant(
    net_config=NET_CONFIG["small_os8"],
    scale=1.0,
    stages_pattern=["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    out_index=[0, 2, 7, 10],
)

LARGE_X1_0_OS8 = MobileNetV3Variant(
    net_config=NET_CONFIG["large_os8"],
    scale=1.0,
    stages_pattern=[
        "blocks[0]",
        "blocks[2]",
        "blocks[5]",
        "blocks[11]",
        "blocks[14]",
    ],
    out_index=[2, 5, 11, 14],
)


def get_variant(
    variant: Literal[
        "SMALL_X0_35",
        "SMALL_X0_5",
        "SMALL_X0_75",
        "SMALL_X1_0",
        "SMALL_X1_25",
        "LARGE_X0_35",
        "LARGE_X0_5",
        "LARGE_X0_75",
        "LARGE_X1_0",
        "LARGE_X1_25",
        "SMALL_X1_0_OS8",
        "LARGE_X1_0_OS8",
    ],
) -> MobileNetV3Variant:
    variants = {
        "SMALL_X0_35": SMALL_X0_35,
        "SMALL_X0_5": SMALL_X0_5,
        "SMALL_X0_75": SMALL_X0_75,
        "SMALL_X1_0": SMALL_X1_0,
        "SMALL_X1_25": SMALL_X1_25,
        "LARGE_X0_35": LARGE_X0_35,
        "LARGE_X0_5": LARGE_X0_5,
        "LARGE_X0_75": LARGE_X0_75,
        "LARGE_X1_0": LARGE_X1_0,
        "LARGE_X1_25": LARGE_X1_25,
        "SMALL_X1_0_OS8": SMALL_X1_0_OS8,
        "LARGE_X1_0_OS8": LARGE_X1_0_OS8,
    }
    if variant not in variants:  # pragma: no cover
        raise ValueError(
            "MicroNet model variant should be in "
            f"{list(variants.keys())}, got {variant}."
        )
    return variants[variant]
