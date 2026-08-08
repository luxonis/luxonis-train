"""Executable shape contract cases for the backbone nodes.

Every concrete module under `luxonis_train.nodes.backbones` that owns a
`forward` method is listed in `CASES`. Each factory builds the module,
the tensors to feed it and the bindings the documented math symbols take
for that particular configuration.
"""

from collections.abc import Callable
from typing import cast
from unittest.mock import patch

import torch
from torch import Size, Tensor, nn

from luxonis_train.nodes.backbones.contextspatial import (
    ContextPath,
    ContextSpatial,
    SpatialPath,
)
from luxonis_train.nodes.backbones.ddrnet import DDRNet
from luxonis_train.nodes.backbones.ddrnet.blocks import (
    DAPPM,
    DAPPMBranch,
    MergeDAPPMBranch,
)
from luxonis_train.nodes.backbones.dinov3 import DinoV3
from luxonis_train.nodes.backbones.dinov3.rope_position_encoding import (
    RopePositionEmbedding,
)
from luxonis_train.nodes.backbones.efficientnet import EfficientNet
from luxonis_train.nodes.backbones.efficientrep import EfficientRep
from luxonis_train.nodes.backbones.efficientvit import EfficientViT
from luxonis_train.nodes.backbones.efficientvit.blocks import (
    DepthWiseSeparableConv,
    EfficientViTBlock,
    LightweightMLABlock,
    MobileBottleneckBlock,
)
from luxonis_train.nodes.backbones.ghostfacenet import GhostFaceNet
from luxonis_train.nodes.backbones.ghostfacenet.blocks import (
    AttentionGhostModuleV2,
    GhostBottleneckLayer,
    GhostBottleneckV2,
    OriginalGhostModuleV2,
)
from luxonis_train.nodes.backbones.ghostfacenet.ghostfacenet import (
    LayerParamsDict as GhostLayerParamsDict,
)
from luxonis_train.nodes.backbones.micronet import MicroNet
from luxonis_train.nodes.backbones.micronet.blocks import (
    ChannelShuffle,
    DepthSpatialSepConv,
    DYShiftMax,
    MicroBlock,
    SpatialSepConvSF,
    Stem,
)
from luxonis_train.nodes.backbones.mobilenetv2 import MobileNetV2
from luxonis_train.nodes.backbones.mobileone import MobileOne
from luxonis_train.nodes.backbones.pplcnet_v3 import PPLCNetV3
from luxonis_train.nodes.backbones.pplcnet_v3.blocks import (
    AffineActivation,
    AffineBlock,
    LCNetV3Block,
    LCNetV3Layer,
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

from .case import ContractCase, case, feature_map, pyramid

IMAGE_SIZE = Size((3, 64, 64))
SMALL_IMAGE_SIZE = Size((3, 32, 32))


class _StubContextBackbone(nn.Module):
    """Feature pyramid stub standing in for the context backbone."""

    def forward(self, x: Tensor) -> list[Tensor]:
        batch, _, height, width = x.shape
        return [
            torch.randn(batch, 8, height // 4, width // 4),
            torch.randn(batch, 16, height // 8, width // 8),
            torch.randn(batch, 24, height // 16, width // 16),
            torch.randn(batch, 32, height // 32, width // 32),
        ]


class _StubHubBackbone(nn.Module):
    """Stub of the EfficientNet model downloaded from Torch Hub."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.act1 = nn.ReLU()
        self.blocks = nn.ModuleList(
            [nn.Conv2d(4, 4, 3, padding=1) for _ in range(3)]
        )


class _StubDinoBackbone(nn.Module):
    """Stub of the DINOv3 model downloaded from Torch Hub."""

    def __init__(self) -> None:
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


def _context_spatial() -> ContractCase:
    module = ContextSpatial(
        context_backbone=_StubContextBackbone(),
        in_sizes=IMAGE_SIZE,
    )
    inputs = {"inputs": torch.randn(2, 3, 64, 64)}
    with torch.no_grad():
        module(**inputs)
    return case(
        module,
        inputs,
        {"B": 2, "C_in": 3, "H": 64, "W": 64},
        key="features",
    )


def _context_path() -> ContractCase:
    module = ContextPath(_StubContextBackbone())
    inputs = {"x": torch.randn(2, 3, 64, 64)}
    with torch.no_grad():
        module(**inputs)
    return case(
        module,
        inputs,
        {"B": 2, "C_in": 3, "H": 64, "W": 64},
        key="features",
    )


def _merge_dappm_branch() -> ContractCase:
    return case(
        MergeDAPPMBranch(
            in_channels=8,
            kernel_size=3,
            stride=2,
            branch_channels=4,
        ),
        {
            "x": torch.randn(2, 8, 16, 16),
            "skip_x": torch.randn(2, 4, 16, 16),
        },
        {"B": 2, "C_in": 8, "H": 16, "W": 16, "branch_channels": 4},
    )


def _ddrnet() -> ContractCase:
    module = DDRNet(
        channels=4,
        high_resolution_channels=8,
        use_aux_heads=True,
        spp_width=4,
        spp_kernel_sizes=[1, 3, 0],
        spp_strides=[1, 2, 0],
        layer3_repeats=2,
        layers=[1, 1, 1, 1, 1, 1, 1, 1],
        in_sizes=IMAGE_SIZE,
    )
    return pyramid(
        module,
        {"inputs": torch.randn(2, 3, 64, 64)},
        [(8, 8, 8), (16, 8, 8)],
    )


def _dinov3() -> list[ContractCase]:
    with patch.object(torch.hub, "load", lambda **_: _StubDinoBackbone()):
        dense = DinoV3(
            weights_link="weights.pth",
            depth=2,
            original_in_shape=SMALL_IMAGE_SIZE,
            in_sizes=SMALL_IMAGE_SIZE,
        )
        sequence = DinoV3(
            weights_link="weights.pth",
            return_sequence=True,
            original_in_shape=SMALL_IMAGE_SIZE,
            in_sizes=SMALL_IMAGE_SIZE,
        )
    return [
        case(
            dense,
            {"inputs": torch.randn(2, 3, 32, 32)},
            {
                "B": 2,
                "C_in": 3,
                "H": 32,
                "W": 32,
                "C_out": 8,
                "depth": 2,
                "patch": 16,
            },
            key="features",
        ),
        case(
            sequence,
            {"inputs": torch.randn(2, 3, 32, 32)},
            {
                "B": 2,
                "C_in": 3,
                "H": 32,
                "W": 32,
                "C_out": 1,
                "patch": 16,
            },
            key="features",
            mode="sequence",
        ),
    ]


def _rope_position_embedding() -> ContractCase:
    module = RopePositionEmbedding(8, num_heads=1)
    module.eval()
    with torch.no_grad():
        sin, cos = module(H=4, W=6)
    return ContractCase(
        module=module,
        inputs={"H": 4, "W": 6},
        bindings={"H": 4, "W": 6, "head_dim": 8},
        outputs={"sin": sin, "cos": cos},
    )


def _efficientnet() -> ContractCase:
    with patch.object(torch.hub, "load", lambda *_, **__: _StubHubBackbone()):
        module = EfficientNet(out_indices=[0, 2], weights="none")
    return pyramid(
        module,
        {"inputs": torch.randn(2, 3, 32, 32)},
        [(4, 16, 16), (4, 16, 16)],
    )


def _efficientrep() -> list[ContractCase]:
    shapes = [(8, 16, 16), (16, 8, 8), (16, 4, 4), (16, 2, 2)]
    return [
        pyramid(
            EfficientRep(
                channels_list=[8, 8, 16, 16, 16],
                n_repeats=[1, 2, 2, 2, 2],
                depth_multiplier=1.0,
                width_multiplier=1.0,
                block=block,
                weights="none",
                in_sizes=IMAGE_SIZE,
            ),
            {"inputs": torch.randn(2, 3, 64, 64)},
            shapes,
        )
        for block in ("RepBlock", "CSPStackRepBlock")
    ]


def _efficientvit() -> ContractCase:
    module = EfficientViT(
        width_list=[8, 8, 16, 32, 64],
        depth_list=[1, 1, 1, 1, 1],
        dim=8,
        expand_ratio=2,
        in_sizes=IMAGE_SIZE,
    )
    return pyramid(
        module,
        {"x": torch.randn(2, 3, 64, 64)},
        [
            (8, 32, 32),
            (8, 16, 16),
            (16, 8, 8),
            (32, 4, 4),
            (64, 2, 2),
        ],
    )


GHOST_LAYER_PARAMS: list[GhostLayerParamsDict] = [
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


def _ghostfacenet() -> ContractCase:
    module = GhostFaceNet(
        width_multiplier=1,
        layer_params=GHOST_LAYER_PARAMS,
        in_sizes=IMAGE_SIZE,
    )
    module.initialize_weights("none")
    return pyramid(
        module,
        {"x": torch.randn(2, 3, 64, 64)},
        [
            (16, 32, 32),
            (16, 32, 32),
            (24, 16, 16),
            (24, 16, 16),
        ],
    )


def _micronet() -> ContractCase:
    module = MicroNet(**MicroNet.get_variants()[1]["M1"])
    return pyramid(
        module,
        {"inputs": torch.randn(2, 3, 64, 64)},
        [
            (8, 16, 16),
            (16, 8, 8),
            (32, 4, 4),
            (576, 2, 2),
        ],
    )


def _mobilenet_v2() -> ContractCase:
    return pyramid(
        MobileNetV2(out_indices=[0, 2], weights="none"),
        {"inputs": torch.randn(2, 3, 32, 32)},
        [(32, 16, 16), (24, 8, 8)],
    )


def _mobileone() -> ContractCase:
    module = MobileOne(
        width_multipliers=(0.25, 0.25, 0.25, 0.25),
        n_conv_branches=1,
        use_se=False,
        in_sizes=IMAGE_SIZE,
    )
    return pyramid(
        module,
        {"inputs": torch.randn(2, 3, 64, 64)},
        [
            (16, 32, 32),
            (16, 16, 16),
            (32, 8, 8),
            (64, 4, 4),
            (128, 2, 2),
        ],
    )


PPLCNET_LAYER_PARAMS: list[PPLCNetLayerParamsDict] = [
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


def _pplcnet_v3() -> list[ContractCase]:
    cases: list[ContractCase] = []
    expected: dict[bool, list[tuple[int, int, int]]] = {
        False: [
            (16, 16, 16),
            (16, 16, 16),
            (16, 8, 8),
            (16, 8, 8),
            (16, 1, 12),
        ],
        True: [
            (16, 16, 16),
            (24, 16, 16),
            (56, 8, 8),
            (480, 8, 8),
        ],
    }
    for use_detection_backbone, shapes in expected.items():
        module = PPLCNetV3(
            scale=1.0,
            n_branches=1,
            use_detection_backbone=use_detection_backbone,
            max_text_len=12,
            layer_params=PPLCNET_LAYER_PARAMS,
            in_sizes=IMAGE_SIZE,
        )
        cases.append(pyramid(module, {"x": torch.randn(2, 3, 64, 64)}, shapes))
    return cases


def _recsubnet() -> ContractCase:
    module = RecSubNet(
        base_channels=4,
        width_multipliers=[1, 2],
        out_channels=3,
        in_sizes=Size((3, 16, 16)),
    )
    return case(
        module,
        {"x": torch.randn(2, 3, 16, 16)},
        {"B": 2, "C_in": 3, "H": 16, "W": 16},
    )


def _repvgg() -> ContractCase:
    module = RepVGG(
        n_blocks=(1, 1, 1, 1),
        width_multiplier=(0.25, 0.25, 0.25, 0.25),
        override_groups_map={0: 1},
        use_se=True,
        in_sizes=IMAGE_SIZE,
    )
    return pyramid(
        module,
        {"inputs": torch.randn(2, 3, 64, 64)},
        [
            (16, 16, 16),
            (32, 8, 8),
            (64, 4, 4),
            (128, 2, 2),
        ],
    )


def _resnet() -> ContractCase:
    return pyramid(
        ResNet(variant="18", weights="none"),
        {"inputs": torch.randn(2, 3, 32, 32)},
        [
            (64, 8, 8),
            (128, 4, 4),
            (256, 2, 2),
            (512, 1, 1),
        ],
    )


def _rexnet() -> ContractCase:
    module = ReXNetV1_lite(
        fix_head_stem=True,
        multiplier=0.25,
        kernel_sizes=[3, 3, 3, 3, 3, 3],
        out_indices=[1, 4, 10, 17],
    )
    return pyramid(
        module,
        {"inputs": torch.randn(2, 3, 64, 64)},
        [
            (16, 32, 32),
            (24, 8, 8),
            (48, 4, 4),
            (1280, 2, 2),
        ],
    )


CASES: dict[
    type[nn.Module], Callable[[], ContractCase | list[ContractCase]]
] = {
    ContextSpatial: _context_spatial,
    ContextPath: _context_path,
    SpatialPath: lambda: case(
        SpatialPath(3, 128),
        {"x": torch.randn(2, 3, 32, 32)},
        {"B": 2, "C_in": 3, "C_out": 128, "H": 32, "W": 32},
    ),
    DAPPMBranch: lambda: case(
        DAPPMBranch(in_channels=8, kernel_size=3, stride=2, branch_channels=4),
        {"x": torch.randn(2, 8, 16, 16)},
        {"B": 2, "C_in": 8, "H": 16, "W": 16, "branch_channels": 4},
    ),
    MergeDAPPMBranch: _merge_dappm_branch,
    DAPPM: lambda: case(
        DAPPM(
            in_channels=8,
            branch_channels=4,
            out_channels=6,
            kernel_sizes=[1, 3, 0],
            strides=[1, 2, 0],
        ),
        {"x": torch.randn(2, 8, 16, 16)},
        {"B": 2, "C_in": 8, "C_out": 6, "H": 16, "W": 16},
    ),
    DDRNet: _ddrnet,
    DinoV3: _dinov3,
    RopePositionEmbedding: _rope_position_embedding,
    EfficientNet: _efficientnet,
    EfficientRep: _efficientrep,
    DepthWiseSeparableConv: lambda: feature_map(
        DepthWiseSeparableConv(8, 6, kernel_size=3, stride=2),
        torch.randn(2, 8, 16, 16),
        out_channels=6,
        out_size=(8, 8),
    ),
    MobileBottleneckBlock: lambda: feature_map(
        MobileBottleneckBlock(8, 6, stride=2, expand_ratio=2),
        torch.randn(2, 8, 16, 16),
        out_channels=6,
        out_size=(8, 8),
    ),
    LightweightMLABlock: lambda: feature_map(
        LightweightMLABlock(
            8,
            6,
            n_heads=1,
            dimension=2,
            scale_factors=(),
            use_residual=False,
        ),
        torch.randn(2, 8, 8, 8),
        out_channels=6,
    ),
    EfficientViTBlock: lambda: feature_map(
        EfficientViTBlock(
            n_channels=8,
            head_dim=4,
            expansion_factor=2,
            aggregation_scales=(3,),
        ),
        torch.randn(2, 8, 8, 8),
    ),
    EfficientViT: _efficientvit,
    OriginalGhostModuleV2: lambda: feature_map(
        OriginalGhostModuleV2(8, 6), torch.randn(2, 8, 8, 8), out_channels=6
    ),
    AttentionGhostModuleV2: lambda: feature_map(
        AttentionGhostModuleV2(8, 6), torch.randn(2, 8, 8, 8), out_channels=6
    ),
    GhostBottleneckV2: lambda: feature_map(
        GhostBottleneckV2(
            in_channels=8,
            hidden_channels=12,
            out_channels=6,
            kernel_size=3,
            stride=2,
            se_ratio=0.25,
            mode="attention",
        ),
        torch.randn(2, 8, 8, 8),
        out_channels=6,
        out_size=(4, 4),
    ),
    GhostBottleneckLayer: lambda: feature_map(
        GhostBottleneckLayer(
            width_multiplier=1,
            input_channel=8,
            kernel_sizes=[3],
            expand_sizes=[16],
            output_channels=[16],
            se_ratios=[0.0],
            strides=[2],
            mode="original",
        ),
        torch.randn(2, 8, 8, 8),
        out_channels=16,
        out_size=(4, 4),
    ),
    GhostFaceNet: _ghostfacenet,
    MicroBlock: lambda: feature_map(
        MicroBlock(8, 8, groups_1=(0, 4), groups_2=(2, 2)),
        torch.randn(2, 8, 8, 8),
        argument="inputs",
    ),
    ChannelShuffle: lambda: feature_map(
        ChannelShuffle(2), torch.randn(2, 8, 8, 8)
    ),
    DYShiftMax: lambda: feature_map(
        DYShiftMax(8, 8, groups=2), torch.randn(2, 8, 8, 8)
    ),
    SpatialSepConvSF: lambda: feature_map(
        SpatialSepConvSF(3, (4, 4), 3, 2),
        torch.randn(2, 3, 16, 16),
        out_channels=16,
        out_size=(8, 8),
    ),
    Stem: lambda: feature_map(
        Stem(3, 2, (4, 4)),
        torch.randn(2, 3, 16, 16),
        out_channels=16,
        out_size=(8, 8),
    ),
    DepthSpatialSepConv: lambda: feature_map(
        DepthSpatialSepConv(8, (2, 2), 3, 2),
        torch.randn(2, 8, 16, 16),
        out_channels=32,
        out_size=(8, 8),
    ),
    MicroNet: _micronet,
    MobileNetV2: _mobilenet_v2,
    MobileOne: _mobileone,
    AffineActivation: lambda: case(
        AffineActivation(), {"x": torch.randn(2, 4)}, {"S": (2, 4)}
    ),
    AffineBlock: lambda: case(
        AffineBlock(), {"x": torch.randn(2, 4)}, {"S": (2, 4)}
    ),
    LCNetV3Block: lambda: feature_map(
        LCNetV3Block(in_channels=8, out_channels=6, kernel_size=3, stride=2),
        torch.randn(2, 8, 16, 16),
        out_channels=6,
        out_size=(8, 8),
    ),
    LCNetV3Layer: lambda: feature_map(
        LCNetV3Layer(
            in_channels=8,
            out_channels=[16, 16],
            kernel_sizes=[3, 3],
            strides=[2, 1],
            use_se=[False, True],
        ),
        torch.randn(2, 8, 16, 16),
        out_channels=16,
        out_size=(8, 8),
    ),
    PPLCNetV3: _pplcnet_v3,
    RecSubNet: _recsubnet,
    RepVGG: _repvgg,
    ResNet: _resnet,
    ReXNetV1_lite: _rexnet,
    LinearBottleneck: lambda: feature_map(
        LinearBottleneck(8, 16, t=6, stride=2),
        torch.randn(2, 8, 8, 8),
        out_channels=16,
        out_size=(4, 4),
    ),
}

UNSUPPORTED: dict[type[nn.Module], str] = {}


def _register_quantized_affine_block() -> None:
    """Register the quantized block when `aimet_torch` is installed."""
    try:
        from luxonis_train.nodes.backbones.pplcnet_v3.blocks import (
            QuantizedAffineBlock,
        )
    except ImportError:  # pragma: no cover - depends on the environment
        return

    def factory() -> ContractCase:
        module = QuantizedAffineBlock.from_module(AffineBlock())
        for name in ("input_quantizers", "output_quantizers"):
            cast(nn.ModuleList, getattr(module, name))[0] = nn.Identity()
        return case(module, {"x": torch.randn(2, 4)}, {"S": (2, 4)})

    CASES[QuantizedAffineBlock] = factory


_register_quantized_affine_block()
