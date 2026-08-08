from collections.abc import Sequence
from typing import Literal, cast
from unittest.mock import Mock

import pytest
import torch
from torch import Size, Tensor, nn

from luxonis_train.nodes.backbones.contextspatial import (
    ContextPath,
    ContextSpatial,
    SpatialPath,
)
from luxonis_train.nodes.backbones.ddrnet import DDRNet
from luxonis_train.nodes.backbones.ddrnet.blocks import make_layer
from luxonis_train.nodes.backbones.dinov3 import DinoV3
from luxonis_train.nodes.backbones.dinov3.dinov3 import (
    TransformerBackboneReturnsIntermediateLayers,
)
from luxonis_train.nodes.backbones.dinov3.rope_position_encoding import (
    RopePositionEmbedding,
)
from luxonis_train.nodes.backbones.efficientnet import EfficientNet
from luxonis_train.nodes.backbones.efficientrep import EfficientRep
from luxonis_train.nodes.backbones.efficientvit import EfficientViT
from luxonis_train.nodes.backbones.efficientvit.blocks import (
    LightweightMLABlock,
    MobileBottleneckBlock,
)
from luxonis_train.nodes.backbones.ghostfacenet import GhostFaceNet
from luxonis_train.nodes.backbones.ghostfacenet.ghostfacenet import (
    LayerParamsDict as GhostLayerParamsDict,
)
from luxonis_train.nodes.backbones.micronet import MicroNet
from luxonis_train.nodes.backbones.micronet.blocks import (
    DYShiftMax,
    MicroBlock,
    _make_divisible,
)
from luxonis_train.nodes.backbones.mobilenetv2 import MobileNetV2
from luxonis_train.nodes.backbones.mobileone import MobileOne
from luxonis_train.nodes.backbones.pplcnet_v3 import PPLCNetV3
from luxonis_train.nodes.backbones.pplcnet_v3.blocks import (
    AffineBlock,
    QuantizedAffineBlock,
    scale_up,
)
from luxonis_train.nodes.backbones.pplcnet_v3.pplcnet_v3 import (
    LayerParamsDict as PPLCNetLayerParamsDict,
)
from luxonis_train.nodes.backbones.recsubnet import RecSubNet
from luxonis_train.nodes.backbones.repvgg import RepVGG
from luxonis_train.nodes.backbones.resnet import ResNet
from luxonis_train.nodes.backbones.rexnetv1 import (
    LinearBottleneck,
    ReXNetV1_lite,
)
from luxonis_train.nodes.blocks import ResNetBlock
from luxonis_train.registry import NODES

from .test_shape_contracts import _assert_contract

IN_SIZE = Size((3, 64, 64))


def _assert_pyramid(
    module_type: type[nn.Module],
    inputs: dict[str, Tensor],
    outputs: list[Tensor],
    shapes: list[tuple[int, int, int]],
) -> None:
    """Check a backbone returning a feature pyramid."""
    batch, channels, height, width = next(iter(inputs.values())).shape
    _assert_contract(
        module_type,
        inputs,
        {"features": outputs},
        {
            "B": batch,
            "C_in": channels,
            "H_in": height,
            "W_in": width,
            "N": len(shapes),
            "C": tuple(shape[0] for shape in shapes),
            "H": tuple(shape[1] for shape in shapes),
            "W": tuple(shape[2] for shape in shapes),
        },
    )


def test_dinov3_backbone_interface_requires_implementation():
    backbone = TransformerBackboneReturnsIntermediateLayers()
    with pytest.raises(
        NotImplementedError,
        match="must implement get_intermediate_layers",
    ):
        backbone.get_intermediate_layers(
            torch.randn(1, 3, 16, 16),
            n=1,
            norm=True,
            return_class_token=False,
        )


class _ContextBackbone(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def forward(self, x: Tensor) -> list[Tensor]:
        return [
            x,
            torch.randn(x.shape[0], 8, 16, 16),
            torch.randn(x.shape[0], 16, 8, 8),
            torch.randn(x.shape[0], 24, 4, 4),
            torch.randn(x.shape[0], 32, 2, 2),
        ]


def test_context_spatial_paths(monkeypatch: pytest.MonkeyPatch):
    x = torch.randn(2, 3, 64, 64)
    image: dict[str, int | Sequence[int]] = {
        "B": 2,
        "C_in": 3,
        "H": 64,
        "W": 64,
    }

    spatial = SpatialPath(3, 128)
    _assert_contract(
        SpatialPath,
        {"x": x},
        {"output": spatial(x)},
        image | {"C_out": 128},
    )

    context = ContextPath(_ContextBackbone())
    for _ in range(2):
        # The second call reuses the refinement blocks that the first
        # one created lazily.
        _assert_contract(
            ContextPath, {"x": x}, {"features": context(x)}, dict(image)
        )

    node = ContextSpatial(
        context_backbone=_ContextBackbone(),
        in_sizes=IN_SIZE,
    )
    _assert_contract(
        ContextSpatial, {"inputs": x}, {"features": node(x)}, dict(image)
    )

    monkeypatch.setattr(NODES, "get", lambda _: _ContextBackbone)
    from_registry = ContextSpatial(
        context_backbone="Dummy",
        backbone_kwargs={"unused": True},
        in_sizes=IN_SIZE,
    )
    assert isinstance(from_registry.context_path.backbone, _ContextBackbone)


class _HubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.act1 = nn.ReLU()
        self.blocks = nn.ModuleList([nn.Conv2d(4, 4, 3, padding=1)] * 3)


def test_efficientnet(monkeypatch: pytest.MonkeyPatch):
    load = Mock(return_value=_HubBackbone())
    monkeypatch.setattr(torch.hub, "load", load)

    node = EfficientNet(out_indices=[0, 2], weights="download")
    x = torch.randn(2, 3, 32, 32)

    _assert_pyramid(
        EfficientNet,
        {"inputs": x},
        node(x),
        [(4, 16, 16), (4, 16, 16)],
    )
    load.assert_called_once_with(
        "rwightman/gen-efficientnet-pytorch",
        "efficientnet_lite0",
        pretrained=True,
        trust_repo=True,
    )


@pytest.mark.parametrize("block", ["RepBlock", "CSPStackRepBlock"])
def test_efficientrep(block: Literal["RepBlock", "CSPStackRepBlock"]):
    node = EfficientRep(
        channels_list=[8, 8, 16, 16, 16],
        n_repeats=[1, 2, 2, 2, 2],
        depth_multiplier=1.0,
        width_multiplier=1.0,
        block=block,
        weights="none",
        in_sizes=IN_SIZE,
    ).eval()

    x = torch.randn(2, 3, 64, 64)
    _assert_pyramid(
        EfficientRep,
        {"inputs": x},
        node(x),
        [(8, 16, 16), (16, 8, 8), (16, 4, 4), (16, 2, 2)],
    )
    node._variant = "n"
    assert "efficientrep_" in node.get_weights_url()
    assert EfficientRep.get_variants()[0] == "n"


def test_efficientvit():
    node = EfficientViT(
        width_list=[8, 8, 16, 32, 64],
        depth_list=[1, 1, 1, 1, 1],
        dim=8,
        expand_ratio=2,
        in_sizes=IN_SIZE,
    ).eval()

    x = torch.randn(2, 3, 64, 64)
    _assert_pyramid(
        EfficientViT,
        {"x": x},
        node(x),
        [
            (8, 32, 32),
            (8, 16, 16),
            (16, 8, 8),
            (32, 4, 4),
            (64, 2, 2),
        ],
    )
    assert EfficientViT.get_variants()[0] == "n"


def test_efficientvit_block_options():
    x = torch.randn(2, 8, 4, 4)
    preserving = {
        "B": 2,
        "C_in": 8,
        "C_out": 8,
        "H": 4,
        "W": 4,
        "H_out": 4,
        "W_out": 4,
    }
    block = MobileBottleneckBlock(
        8,
        8,
        expand_ratio=2,
        use_residual=True,
    ).eval()
    _assert_contract(
        MobileBottleneckBlock,
        {"x": x},
        {"output": block(x)},
        dict(preserving),
    )

    attention = LightweightMLABlock(
        8,
        8,
        n_heads=1,
        dimension=2,
        scale_factors=(),
        use_residual=False,
    ).eval()
    _assert_contract(
        LightweightMLABlock,
        {"x": x},
        {"output": attention(x)},
        dict(preserving),
    )

    linear_input = torch.randn(1, 6, 2, 3, dtype=torch.float16)
    assert attention.linear_attention(linear_input).dtype == torch.float32
    bfloat_linear = torch.randn(1, 6, 2, 3, dtype=torch.bfloat16)
    assert attention.linear_attention(bfloat_linear).dtype == torch.float32
    quadratic_input = torch.randn(1, 6, 1, 2, dtype=torch.bfloat16)
    assert attention.quadratic_attention(quadratic_input).shape == (
        1,
        2,
        1,
        2,
    )


def test_ghostfacenet():
    params: list[GhostLayerParamsDict] = [
        {
            "mode": "original",
            "kernel_sizes": [3],
            "expand_sizes": [16],
            "output_channels": [16],
            "se_ratios": [0.0],
            "strides": [1],
        },
        {
            "mode": "attention",
            "kernel_sizes": [3],
            "expand_sizes": [24],
            "output_channels": [24],
            "se_ratios": [0.25],
            "strides": [2],
        },
    ]
    node = GhostFaceNet(
        width_multiplier=1,
        layer_params=params,
        in_sizes=IN_SIZE,
    ).eval()
    node.initialize_weights("none")

    x = torch.randn(2, 3, 64, 64)
    _assert_pyramid(
        GhostFaceNet,
        {"x": x},
        node(x),
        [(16, 32, 32), (16, 32, 32), (24, 16, 16), (24, 16, 16)],
    )
    assert GhostFaceNet.get_variants()[0] == "V2"


def test_micronet():
    params = MicroNet.get_variants()[1]["M1"]
    node = MicroNet(**params).eval()

    x = torch.randn(2, 3, 64, 64)
    _assert_pyramid(
        MicroNet,
        {"inputs": x},
        node(x),
        [(8, 16, 16), (16, 8, 8), (32, 4, 4), (576, 2, 2)],
    )
    assert MicroNet.get_variants()[0] == "M1"


def test_micronet_block_edges():
    x = torch.randn(2, 8, 8, 8)
    block = MicroBlock(
        8,
        8,
        groups_1=(0, 4),
        groups_2=(2, 2),
    ).eval()
    _assert_contract(
        MicroBlock,
        {"inputs": x},
        {"output": block(x)},
        {
            "B": 2,
            "C_in": 8,
            "C_out": 8,
            "H": 8,
            "W": 8,
            "H_out": 8,
            "W_out": 8,
        },
    )

    invalid = DYShiftMax(8, 8, groups=2)
    invalid.exp = 3  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="Expansion"):
        invalid(x)

    assert _make_divisible(9, 8) == 16


def test_torchvision_backbones():
    x = torch.randn(2, 3, 64, 64)

    mobile = MobileNetV2(out_indices=[0, 2], weights="none").eval()
    _assert_pyramid(
        MobileNetV2,
        {"inputs": x},
        mobile(x),
        [(32, 32, 32), (24, 16, 16)],
    )

    resnet = ResNet(variant="18", weights="none").eval()
    _assert_pyramid(
        ResNet,
        {"inputs": x},
        resnet(x),
        [(64, 16, 16), (128, 8, 8), (256, 4, 4), (512, 2, 2)],
    )
    assert ResNet.get_variants()[0] == "18"

    with pytest.raises(ValueError, match="variant"):
        ResNet._get_backbone("invalid")  # type: ignore[arg-type]


def test_mobileone():
    node = MobileOne(
        width_multipliers=(0.25, 0.25, 0.25, 0.25),
        n_conv_branches=1,
        use_se=False,
        in_sizes=IN_SIZE,
    ).eval()
    x = torch.randn(2, 3, 64, 64)
    _assert_pyramid(
        MobileOne,
        {"inputs": x},
        node(x),
        [
            (16, 32, 32),
            (16, 16, 16),
            (32, 8, 8),
            (64, 4, 4),
            (128, 2, 2),
        ],
    )
    assert MobileOne.get_variants()[0] == "s0"

    with pytest.raises(ValueError, match="cannot exceed"):
        node._make_stage(8, n_blocks=1, n_se_blocks=2)
    assert len(node._make_stage(8, n_blocks=1, n_se_blocks=1)) == 2


PPLCN_PARAMS: list[PPLCNetLayerParamsDict] = [
    {
        "kernel_sizes": [3],
        "out_channels": [8],
        "strides": [1],
        "use_se": [False],
    },
    {
        "kernel_sizes": [3, 3],
        "out_channels": [8, 8],
        "strides": [2, 1],
        "use_se": [False, False],
    },
    {
        "kernel_sizes": [3],
        "out_channels": [8],
        "strides": [1],
        "use_se": [True],
    },
    {
        "kernel_sizes": [3],
        "out_channels": [8],
        "strides": [2],
        "use_se": [False],
    },
    {
        "kernel_sizes": [3],
        "out_channels": [8],
        "strides": [1],
        "use_se": [False],
    },
]


@pytest.mark.parametrize("use_detection_backbone", [False, True])
def test_pplcnet_v3(use_detection_backbone: bool):
    node = PPLCNetV3(
        scale=1.0,
        n_branches=1,
        use_detection_backbone=use_detection_backbone,
        max_text_len=12,
        layer_params=PPLCN_PARAMS,
        in_sizes=IN_SIZE,
    ).eval()

    x = torch.randn(2, 3, 64, 64)
    expected = (
        [
            (16, 16, 16),
            (24, 16, 16),
            (56, 8, 8),
            (480, 8, 8),
        ]
        if use_detection_backbone
        else [
            (16, 16, 16),
            (16, 16, 16),
            (16, 8, 8),
            (16, 8, 8),
            (16, 1, 12),
        ]
    )
    _assert_pyramid(PPLCNetV3, {"x": x}, node(x), expected)
    assert PPLCNetV3.get_variants()[0] == "rec-light"


def test_quantized_affine_block_and_scale_up():
    block = QuantizedAffineBlock.from_module(AffineBlock())
    cast(nn.ModuleList, block.input_quantizers)[0] = nn.Identity()
    cast(nn.ModuleList, block.output_quantizers)[0] = nn.Identity()

    x = torch.randn(2, 4)
    torch.testing.assert_close(block(x), x)
    assert scale_up(18, 1.0) == 32


def test_recsubnet():
    node = RecSubNet(
        base_channels=4,
        width_multipliers=[1, 2],
        out_channels=3,
        in_sizes=IN_SIZE,
    ).eval()

    x = torch.randn(2, 3, 16, 16)
    output = node(x)
    _assert_contract(
        RecSubNet,
        {"x": x},
        output,
        {"B": 2, "C_in": 3, "H": 16, "W": 16},
    )
    assert output["original"] is x
    assert RecSubNet.get_variants()[0] == "l"


def test_repvgg():
    node = RepVGG(
        n_blocks=(1, 1, 1, 1),
        width_multiplier=(0.25, 0.25, 0.25, 0.25),
        override_groups_map={0: 1},
        use_se=True,
        in_sizes=IN_SIZE,
    ).eval()

    x = torch.randn(2, 3, 64, 64)
    _assert_pyramid(
        RepVGG,
        {"inputs": x},
        node(x),
        [(16, 16, 16), (32, 8, 8), (64, 4, 4), (128, 2, 2)],
    )
    assert RepVGG.get_variants()[0] == "A0"


def test_rexnet_and_linear_bottlenecks():
    x = torch.randn(2, 3, 64, 64)
    node = ReXNetV1_lite(
        fix_head_stem=True,
        multiplier=0.25,
        kernel_sizes=[3, 3, 3, 3, 3, 3],
        out_indices=[1, 4, 10, 17],
    ).eval()
    _assert_pyramid(
        ReXNetV1_lite,
        {"inputs": x},
        node(x),
        [(16, 32, 32), (24, 8, 8), (48, 4, 4), (1280, 2, 2)],
    )

    features = torch.randn(2, 8, 8, 8)
    _assert_contract(
        LinearBottleneck,
        {"x": features},
        {"output": LinearBottleneck(8, 8, t=1)(features)},
        {
            "B": 2,
            "C_in": 8,
            "C_out": 8,
            "H": 8,
            "W": 8,
            "H_out": 8,
            "W_out": 8,
        },
    )
    _assert_contract(
        LinearBottleneck,
        {"x": features},
        {"output": LinearBottleneck(8, 16, t=6, stride=2)(features)},
        {
            "B": 2,
            "C_in": 8,
            "C_out": 16,
            "H": 8,
            "W": 8,
            "H_out": 4,
            "W_out": 4,
        },
    )


@pytest.mark.parametrize("use_aux_heads", [False, True])
def test_ddrnet(use_aux_heads: bool):
    node = DDRNet(
        channels=4,
        high_resolution_channels=8,
        use_aux_heads=use_aux_heads,
        spp_width=4,
        spp_kernel_sizes=[1, 3, 0],
        spp_strides=[1, 2, 0],
        layer3_repeats=2,
        layers=[1, 1, 1, 1, 1, 1, 1, 1],
        in_sizes=IN_SIZE,
    ).eval()
    node.initialize_weights()

    x = torch.randn(2, 3, 64, 64)
    expected = [(16, 8, 8)]
    if use_aux_heads:
        expected.insert(0, (8, 8, 8))
    _assert_pyramid(DDRNet, {"inputs": x}, node(x), expected)

    node._variant = None
    with pytest.raises(ValueError, match="predefined variant"):
        node.get_weights_url()
    node._variant = "23-slim"
    assert node.get_weights_url().endswith("ddrnet_23slim_coco.ckpt")
    assert DDRNet.get_variants()[0] == "23-slim"


def test_make_ddr_layer_with_multiple_blocks():
    layer = make_layer(
        ResNetBlock,
        in_channels=8,
        channels=8,
        n_blocks=3,
    ).eval()
    assert layer(torch.randn(2, 8, 8, 8)).shape == (2, 8, 8, 8)


class _DinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.num_heads = 2
        self.patch_size = 16
        self.rope_embed = RopePositionEmbedding(8, num_heads=2)
        self.parameter = nn.Parameter(torch.ones(1))

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int,
        norm: bool,
        return_class_token: bool,
    ) -> list[Tensor] | list[tuple[Tensor, Tensor]]:
        del norm
        if return_class_token:
            return [
                (
                    torch.zeros(x.shape[0], 4, 8),
                    self.parameter + x[:, 0, 0, 0, None],
                )
            ]
        return [torch.zeros(x.shape[0], 4, 8) for _ in range(n)]


def test_dinov3(monkeypatch: pytest.MonkeyPatch):
    backbones: list[_DinoBackbone] = []

    def load(**_: object) -> _DinoBackbone:
        backbone = _DinoBackbone()
        backbones.append(backbone)
        return backbone

    monkeypatch.setattr(torch.hub, "load", load)
    dense = DinoV3(
        weights_link="weights.pth",
        depth=2,
        original_in_shape=Size((3, 32, 32)),
        in_sizes=Size((3, 32, 32)),
    )
    x = torch.randn(2, 3, 32, 32)
    _assert_contract(
        DinoV3,
        {"inputs": x},
        {"features": dense(x)},
        {
            "B": 2,
            "C_in": 3,
            "H": 32,
            "W": 32,
            "C_out": 8,
            "depth": 2,
            "patch": 16,
        },
    )

    sequence = DinoV3(
        weights_link="weights.pth",
        return_sequence=True,
        freeze_backbone=True,
        original_in_shape=Size((3, 32, 32)),
        in_sizes=Size((3, 32, 32)),
    )
    _assert_contract(
        DinoV3,
        {"inputs": x},
        {"features": sequence(x)},
        {"B": 2, "C_in": 3, "H": 32, "W": 32, "C_out": 1, "patch": 16},
        mode="sequence",
    )
    assert not any(
        parameter.requires_grad for parameter in backbones[-1].parameters()
    )
    assert DinoV3.get_variants()[0] == "vits16"

    with pytest.raises(ValueError, match="Unsupported variant"):
        DinoV3._get_backbone("weights", variant="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("normalize_coords", ["min", "max", "separate"])
def test_rope_position_embedding(normalize_coords: str):
    rope = RopePositionEmbedding(
        8,
        num_heads=2,
        base=None,
        min_period=1.0,
        max_period=10.0,
        normalize_coords=normalize_coords,  # type: ignore[arg-type]
        shift_coords=0.1,
        jitter_coords=1.1,
        rescale_coords=1.2,
    )
    sin, cos = rope(H=2, W=3)
    _assert_contract(
        RopePositionEmbedding,
        {"H": 2, "W": 3},
        {"sin": sin, "cos": cos},
        {"H": 2, "W": 3, "head_dim": 4},
    )


def test_rope_position_embedding_errors():
    with pytest.raises(ValueError, match="Either"):
        RopePositionEmbedding(8, num_heads=2, base=None)
    with pytest.raises(ValueError, match="Either"):
        RopePositionEmbedding(
            8,
            num_heads=2,
            base=100.0,
            min_period=1.0,
            max_period=10.0,
        )

    rope = RopePositionEmbedding(8, num_heads=2)
    rope.normalize_coords = "invalid"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown normalize_coords"):
        rope(H=2, W=2)
