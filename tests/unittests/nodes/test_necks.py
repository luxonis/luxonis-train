from typing import Literal

import pytest
import torch
from torch import Size

from luxonis_train.nodes.necks.reppan_neck import RepPANNeck
from luxonis_train.nodes.necks.svtr_neck import SVTRNeck
from luxonis_train.nodes.necks.svtr_neck.blocks import Attention, SVTRBlock

from .shape_contracts import assert_contract


@pytest.mark.parametrize(
    ("n_heads", "block"),
    [
        (2, "RepBlock"),
        (3, "CSPStackRepBlock"),
        (4, "RepBlock"),
    ],
)
def test_reppan_neck(
    n_heads: Literal[2, 3, 4],
    block: Literal["RepBlock", "CSPStackRepBlock"],
):
    in_sizes = [Size((8, 32 // 2**i, 32 // 2**i)) for i in range(n_heads)]
    node = RepPANNeck(
        n_heads=n_heads,
        channels_list=[8, 8, 8, 8, 8, 8],
        n_repeats=[2, 2, 2, 2],
        depth_multiplier=1.0,
        width_multiplier=1.0,
        block=block,
        e=0.5,
        weights="none",
        in_sizes=in_sizes,
        original_in_shape=Size((3, 64, 64)),
    ).eval()
    inputs = [torch.randn(2, *size) for size in in_sizes]

    outputs = node(inputs)

    output_channels = (4, 8, 8, 8) if n_heads == 4 else (8,) * n_heads
    assert_contract(
        RepPANNeck,
        {"inputs": inputs},
        {"features": outputs},
        {
            "B": 2,
            "N": n_heads,
            "C_in": tuple(size[0] for size in in_sizes),
            "C_out": output_channels,
            "H": tuple(size[1] for size in in_sizes),
            "W": tuple(size[2] for size in in_sizes),
        },
    )
    node._variant = "n"
    assert node.get_weights_url().endswith("reppanneck_n_coco.ckpt")
    assert RepPANNeck.get_variants()[0] == "n"


def test_reppan_neck_validation():
    with pytest.raises(ValueError, match="Expected width and height"):
        RepPANNeck(
            n_heads=2,
            in_sizes=[Size((8, 15, 15)), Size((8, 8, 8))],
            original_in_shape=Size((3, 63, 63)),
            weights="none",
        )

    node = RepPANNeck(
        n_heads=2,
        channels_list=[8, 8, 8, 8, 8, 8],
        n_repeats=[1, 1, 1, 1],
        width_multiplier=1.0,
        in_sizes=[Size((8, 16, 16)), Size((8, 8, 8))],
        original_in_shape=Size((3, 64, 64)),
        weights="none",
    )
    node.n_heads = 5
    with pytest.raises(ValueError, match="not supported"):
        node._fit_to_n_heads([8] * 6, [1] * 4)


@pytest.mark.parametrize(
    ("mixer", "use_guide", "prenorm"),
    [
        ("global", False, False),
        ("local", True, True),
        ("conv", False, False),
    ],
)
def test_svtr_neck(
    mixer: Literal["global", "local", "conv"],
    use_guide: bool,
    prenorm: bool,
):
    node = SVTRNeck(
        dims=8,
        depth=1,
        mid_channels=8,
        use_guide=use_guide,
        n_heads=2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=0.1,
        mixer=mixer,
        height=4,
        width=4,
        prenorm=prenorm,
        in_sizes=Size((16, 4, 4)),
    ).eval()
    node.initialize_weights("yolo")

    x = torch.randn(2, 16, 4, 4)
    assert_contract(
        SVTRNeck,
        {"x": x},
        {"output": node(x)},
        {"B": 2, "C_in": 16, "C_out": 8, "H": 4, "W": 4},
    )


def test_svtr_attention_options():
    x = torch.randn(2, 16, 8)
    global_attention = Attention(
        8,
        n_heads=2,
        mixer="global",
        kernel_size=3,
        qk_scale=0.5,
    )
    assert_contract(
        Attention,
        {"x": x},
        {"output": global_attention(x)},
        {"B": 2, "L": 16, "C": 8},
    )

    with pytest.raises(ValueError, match="Height and width"):
        Attention(8, n_heads=2, mixer="local")
    with pytest.raises(ValueError, match="Height and width"):
        SVTRBlock(8, n_heads=2, mixer="conv")
