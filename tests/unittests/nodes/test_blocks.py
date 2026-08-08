from collections.abc import Callable, Sequence
from typing import Literal

import pytest
import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import (
    DFL,
    AttentionRefinmentBlock,
    BlockRepeater,
    ConvBlock,
    ConvStack,
    CSPStackRepBlock,
    DropPath,
    EfficientDecoupledBlock,
    FeatureFusionBlock,
    GeneralReparametrizableBlock,
    SegProto,
    SpatialPyramidPoolingBlock,
    SqueezeExciteBlock,
    UpscaleOnline,
    autopad,
)
from luxonis_train.nodes.blocks.blocks import BottleRep, PreciseDecoupledBlock
from luxonis_train.nodes.blocks.resnet import ResNetBlock, ResNetBottleneck
from luxonis_train.nodes.blocks.unet import (
    EncoderBlock,
    SimpleDecoder,
    SimpleDecoderBlock,
    SimpleEncoder,
    UNetDecoder,
    UNetDecoderBlock,
    UNetEncoder,
    UpBlock,
)
from luxonis_train.nodes.blocks.utils import forward_gather

from .shape_contracts import assert_contract


def _assert_maps_features(
    module_type: type[nn.Module],
    module: nn.Module,
    x: Tensor,
    *,
    out_channels: int,
    out_size: int | None = None,
) -> None:
    """Check a block turning one square feature map into another.

    The bindings cover every symbol the feature-map contracts use, so
    the same call works for blocks that keep the channel count or the
    spatial size and simply do not mention the corresponding symbol.
    """
    batch, channels, height, width = x.shape
    spatial = height if out_size is None else out_size
    assert_contract(
        module_type,
        {"x": x},
        {"output": module(x)},
        {
            "B": batch,
            "C_in": channels,
            "C_out": out_channels,
            "H": height,
            "W": width,
            "H_out": spatial,
            "W_out": spatial,
        },
    )


@pytest.mark.parametrize(
    ("kernel_size", "padding", "expected"), [(1, 2, 2), (2, None, 1)]
)
def test_autopad_int(kernel_size: int, padding: int | None, expected: int):
    assert autopad(kernel_size, padding) == expected


def test_autopad_tuple():
    assert autopad((2, 4)) == (1, 2)


@pytest.mark.parametrize(
    ("activation", "use_norm", "activation_type"),
    [
        (True, True, nn.ReLU),
        (False, False, nn.Identity),
        (nn.SiLU(), True, nn.SiLU),
    ],
)
def test_conv_block(
    activation: Callable[[Tensor], Tensor] | bool,
    use_norm: bool,
    activation_type: type[nn.Module],
):
    block = ConvBlock(
        3,
        4,
        kernel_size=3,
        padding=1,
        activation=activation,
        use_norm=use_norm,
    ).eval()

    _assert_maps_features(
        ConvBlock, block, torch.randn(2, 3, 8, 8), out_channels=4
    )
    assert isinstance(block.activation, activation_type)
    assert (block.bn is not None) is use_norm


def test_prediction_blocks():
    x = torch.randn(2, 8, 4, 4)
    bindings: dict[str, int | Sequence[int]] = {
        "B": 2,
        "C_in": 8,
        "H": 4,
        "W": 4,
    }

    precise = PreciseDecoupledBlock(8, 8, 8, n_classes=3, reg_max=4)
    features, classes, regressions = precise(x)
    assert_contract(
        PreciseDecoupledBlock,
        {"x": x},
        {
            "features": features,
            "classes": classes,
            "regressions": regressions,
        },
        bindings | {"n_classes": 3, "reg_max": 4},
    )

    efficient = EfficientDecoupledBlock(8, n_classes=3)
    features, classes, regressions = efficient(x)
    assert_contract(
        EfficientDecoupledBlock,
        {"x": x},
        {
            "features": features,
            "classes": classes,
            "regressions": regressions,
        },
        bindings | {"n_classes": 3},
    )

    _assert_maps_features(
        SegProto,
        SegProto(8, mid_channels=4, out_channels=2),
        x,
        out_channels=2,
    )

    distributions = regressions.repeat(1, 4, 1, 1)
    assert_contract(
        DFL,
        {"x": distributions},
        {"output": DFL(reg_max=4)(distributions)},
        {"B": 2, "reg_max": 4, "H": 4, "W": 4},
    )


@pytest.mark.parametrize("hard_sigmoid", [False, True])
def test_squeeze_excite_block(hard_sigmoid: bool):
    block = SqueezeExciteBlock(
        16,
        4,
        hard_sigmoid=hard_sigmoid,
        activation=nn.SiLU(),
    )
    _assert_maps_features(
        SqueezeExciteBlock, block, torch.rand(2, 16, 8, 8), out_channels=16
    )


def test_general_reparametrizable_block_round_trip():
    block = GeneralReparametrizableBlock(
        16,
        16,
        n_branches=2,
        refine_block="se",
    ).eval()
    x = torch.randn(2, 16, 8, 8)
    expected = block(x)

    assert block.name == "GeneralReparametrizableBlock"
    block.reparametrize()
    torch.testing.assert_close(block(x), expected)
    block.reparametrize()

    block.restore()
    torch.testing.assert_close(block(x), expected)
    block.restore()


def test_general_reparametrizable_block_options():
    x = torch.randn(2, 8, 8, 8)
    disabled = GeneralReparametrizableBlock(
        8,
        8,
        use_scale_layer=False,
        activation=False,
    )
    _assert_maps_features(
        GeneralReparametrizableBlock, disabled, x, out_channels=8
    )

    block = GeneralReparametrizableBlock(
        8,
        16,
        stride=2,
        use_scale_layer=False,
        refine_block=nn.Sigmoid(),
        activation=nn.SiLU(),
    ).eval()

    _assert_maps_features(
        GeneralReparametrizableBlock,
        block,
        x,
        out_channels=16,
        out_size=4,
    )
    block.reparametrize()
    _assert_maps_features(
        GeneralReparametrizableBlock,
        block,
        x,
        out_channels=16,
        out_size=4,
    )

    with pytest.raises(ValueError, match="variance and mean"):
        block._postprocess_fused(
            None,
            None,
            torch.ones(16),
            torch.zeros(16),
            torch.ones(16, 8, 3, 3),
            1e-5,
        )


def test_repeated_and_rep_blocks():
    x = torch.randn(2, 8, 8, 8)

    repeater = BlockRepeater(
        ConvBlock,
        n_repeats=2,
        in_channels=8,
        out_channels=16,
        kernel_size=1,
    )
    _assert_maps_features(BlockRepeater, repeater, x, out_channels=16)
    _assert_maps_features(
        ConvStack, ConvStack(8, 8, n_repeats=1), x, out_channels=8
    )
    _assert_maps_features(
        CSPStackRepBlock, CSPStackRepBlock(8, 8, n_blocks=2), x, out_channels=8
    )

    _assert_maps_features(
        BottleRep, BottleRep(8, 8, weight=True), x, out_channels=8
    )
    _assert_maps_features(
        BottleRep, BottleRep(8, 16, weight=False), x, out_channels=16
    )


def test_feature_processing_blocks():
    x = torch.randn(2, 8, 8, 8)

    _assert_maps_features(
        SpatialPyramidPoolingBlock,
        SpatialPyramidPoolingBlock(8, 4),
        x,
        out_channels=4,
    )
    _assert_maps_features(
        AttentionRefinmentBlock,
        AttentionRefinmentBlock(8, 4),
        x,
        out_channels=4,
    )
    assert_contract(
        FeatureFusionBlock,
        {"x1": x, "x2": x},
        {"output": FeatureFusionBlock(16, 8)(x, x)},
        {"B": 2, "C1": 8, "C2": 8, "C_out": 8, "H": 8, "W": 8},
    )
    assert_contract(
        UpscaleOnline,
        {"x": x, "output_height": 12, "output_width": 10},
        {"output": UpscaleOnline()(x, 12, 10)},
        {
            "B": 2,
            "C_in": 8,
            "H": 8,
            "W": 8,
            "output_height": 12,
            "output_width": 10,
        },
    )


def test_drop_path_modes():
    x = torch.ones(4, 3, 2, 2)

    assert torch.equal(DropPath()(x), x)

    drop_path = DropPath(0.5, scale_by_keep=True).eval()
    assert torch.equal(drop_path(x), x)

    drop_path.train()
    _assert_maps_features(DropPath, drop_path, x, out_channels=3)
    assert DropPath(1.0, scale_by_keep=False)(x).count_nonzero() == 0


@pytest.mark.parametrize(
    "block",
    [
        ResNetBlock(8, 8, final_relu=False, droppath_prob=0.1),
        ResNetBlock(8, 16, stride=2),
        ResNetBottleneck(8, 2, final_relu=False, droppath_prob=0.1),
        ResNetBottleneck(8, 4, stride=2),
    ],
)
def test_residual_blocks(block: nn.Module):
    block.eval()
    # The contract leaves the output resolution free, so the effect of
    # the stride on it is checked by hand.
    expected_spatial = (
        4
        if any(
            isinstance(module, nn.Conv2d) and module.stride == (2, 2)
            for module in block.modules()
        )
        else 8
    )
    output = block(torch.randn(2, 8, 8, 8))
    assert output.shape[-2:] == (expected_spatial, expected_spatial)


def test_forward_gather_and_encoders():
    x = torch.randn(2, 3, 16, 16)
    modules = [nn.Conv2d(3, 4, 1), nn.Conv2d(4, 5, 1)]
    outputs = forward_gather(x, modules)
    assert [output.shape for output in outputs] == [
        (2, 4, 16, 16),
        (2, 5, 16, 16),
    ]

    _assert_maps_features(
        EncoderBlock, EncoderBlock(3, 4, 1, max_pool=False), x, out_channels=4
    )
    _assert_maps_features(
        SimpleEncoder,
        SimpleEncoder(3, 4, [1, 2], n_convolutions=1),
        x,
        out_channels=8,
        out_size=4,
    )
    encoder = UNetEncoder(3, 4, [1, 2], n_convolutions=1)
    assert_contract(
        UNetEncoder,
        {"x": x},
        {"features": encoder(x)},
        {
            "B": 2,
            "C_in": 3,
            "H_in": 16,
            "W_in": 16,
            "N": 3,
            "C": (4, 8, 8),
            "H": (16, 8, 4),
            "W": (16, 8, 4),
        },
    )


@pytest.mark.parametrize(
    "upsample_mode",
    ["simple_upsample", "conv_upsample", "conv_transpose"],
)
def test_up_blocks(
    upsample_mode: Literal[
        "simple_upsample", "conv_upsample", "conv_transpose"
    ],
):
    block = UpBlock(
        8,
        4,
        upsample_mode=upsample_mode,
        kernel_size=2,
        use_norm=False,
        align_corners=True,
        activation=False,
    )
    _assert_maps_features(
        UpBlock, block, torch.randn(2, 8, 4, 4), out_channels=4, out_size=8
    )


def test_decoders():
    x = torch.randn(2, 8, 4, 4)

    simple_block = SimpleDecoderBlock(
        8,
        4,
        kernel_size=2,
        use_norm=True,
        align_corners=True,
        upsample_mode="simple_upsample",
        n_repeats=1,
    )
    _assert_maps_features(
        SimpleDecoderBlock, simple_block, x, out_channels=4, out_size=8
    )

    unet_block = UNetDecoderBlock(
        8,
        4,
        kernel_size=2,
        use_norm=True,
        align_corners=True,
        upsample_mode="simple_upsample",
        n_repeats=1,
    )
    skip = torch.randn(2, 8, 8, 8)
    assert_contract(
        UNetDecoderBlock,
        {"x": x, "skip_x": skip},
        {"output": unet_block(x, skip)},
        {"B": 2, "C_in": 8, "C_skip": 8, "C_out": 4, "H": 4, "W": 4},
    )

    simple = SimpleDecoder(
        base_width=4,
        out_channels=3,
        encoder_width_multipliers=[1, 2],
        n_convolutions=1,
    )
    _assert_maps_features(
        SimpleDecoder, simple, x, out_channels=3, out_size=16
    )

    unet = UNetDecoder(
        base_width=4,
        out_channels=3,
        encoder_width_multipliers=[1, 2],
        n_convolutions=1,
    )
    inputs = [
        torch.randn(2, 4, 16, 16),
        torch.randn(2, 8, 8, 8),
        x,
    ]
    # `UNetDecoder.forward` pops the deepest feature off the list it is
    # given, so the documented inputs are kept in a separate list.
    documented_inputs = list(inputs)
    assert_contract(
        UNetDecoder,
        {"inputs": documented_inputs},
        {"output": unet(inputs)},
        {
            "B": 2,
            "N": 3,
            "C": (4, 8, 8),
            "H": (16, 8, 4),
            "W": (16, 8, 4),
            "C_out": 3,
            "H_out": 16,
            "W_out": 16,
        },
    )
