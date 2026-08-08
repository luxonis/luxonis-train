"""Runtime shape contract cases for the head nodes.

The default `Outputs` group of the detection heads and of `FOMOHead`
documents what those heads return while training, so their default case
keeps the module in training mode and carries the outputs it produced
there. Every other head returns the same structure in both modes and is
evaluated.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import Size, nn

from luxonis_train.nodes.heads import (
    BaseHead,
    BiSeNetHead,
    ClassificationHead,
    DDRNetSegmentationHead,
    DiscSubNetHead,
    EfficientBBoxHead,
    EfficientKeypointBBoxHead,
    FOMOHead,
    GhostFaceNetHead,
    OCRCTCHead,
    PrecisionBBoxHead,
    PrecisionSegmentBBoxHead,
    SegmentationHead,
    TransformerClassificationHead,
    TransformerSegmentationHead,
)
from luxonis_train.nodes.heads.base_detection_head import BaseDetectionHead
from luxonis_train.utils import DatasetMetadata

from .case import ContractCase

BATCH = 2
N_CLASSES = 2
N_KEYPOINTS = 2

DETECTION_IMAGE = Size((3, 64, 64))
DETECTION_SIZES = [Size((8, 8, 8)), Size((8, 4, 4))]
DETECTION_ANCHORS = 8 * 8 + 4 * 4
REG_MAX = 4
N_MASKS = 4


def _head_kwargs(
    in_sizes: Size | list[Size],
    *,
    original_in_shape: Size,
    n_classes: int = N_CLASSES,
    n_keypoints: int = N_KEYPOINTS,
) -> dict[str, Any]:
    return {
        "in_sizes": in_sizes,
        "original_in_shape": original_in_shape,
        "n_classes": n_classes,
        "n_keypoints": n_keypoints,
        "dataset_metadata": DatasetMetadata(
            classes={"": {f"class-{i}": i for i in range(n_classes)}},
            n_keypoints={"": n_keypoints},
        ),
    }


def _detection_kwargs() -> dict[str, Any]:
    return _head_kwargs(DETECTION_SIZES, original_in_shape=DETECTION_IMAGE) | {
        "n_heads": 2,
        "conf_thres": 0.5,
        "iou_thres": 0.5,
        "max_det": 10,
    }


def _detection_inputs() -> list[torch.Tensor]:
    return [torch.randn(BATCH, *size) for size in DETECTION_SIZES]


def _detection_bindings(**extra: int) -> dict[str, int | tuple[int, ...]]:
    return {
        "B": BATCH,
        "N": len(DETECTION_SIZES),
        "C": tuple(size[0] for size in DETECTION_SIZES),
        "H": tuple(size[1] for size in DETECTION_SIZES),
        "W": tuple(size[2] for size in DETECTION_SIZES),
        "n_classes": N_CLASSES,
        **extra,
    }


def _segmentation_case(module: nn.Module) -> ContractCase:
    """Build a case for a head decoding one feature map into logits."""
    inputs = torch.randn(BATCH, 16, 8, 8)
    return ContractCase(
        module=module.eval(),
        inputs={"inputs": inputs},
        bindings={
            "B": BATCH,
            "C_in": 16,
            "H": 8,
            "W": 8,
            "H_image": 32,
            "W_image": 32,
            "n_classes": N_CLASSES,
        },
    )


def _segmentation_head() -> ContractCase:
    return _segmentation_case(
        SegmentationHead(
            **_head_kwargs(
                Size((16, 8, 8)), original_in_shape=Size((3, 32, 32))
            )
        )
    )


def _bisenet_head() -> ContractCase:
    return _segmentation_case(
        BiSeNetHead(
            intermediate_channels=8,
            **_head_kwargs(
                Size((16, 8, 8)), original_in_shape=Size((3, 32, 32))
            ),
        )
    )


def _classification_head() -> ContractCase:
    module = ClassificationHead(
        dropout_rate=0.0,
        **_head_kwargs(Size((8, 4, 4)), original_in_shape=Size((3, 32, 32))),
    ).eval()
    return ContractCase(
        module=module,
        inputs={"inputs": torch.randn(BATCH, 8, 4, 4)},
        bindings={
            "B": BATCH,
            "C_in": 8,
            "H": 4,
            "W": 4,
            "n_classes": N_CLASSES,
        },
    )


def _transformer_classification_head() -> ContractCase:
    module = TransformerClassificationHead(
        dropout_rate=0.0,
        **_head_kwargs(Size((8,)), original_in_shape=Size((3, 32, 32))),
    ).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(BATCH, 8)},
        bindings={"B": BATCH, "C_in": 8, "n_classes": N_CLASSES},
    )


def _transformer_segmentation_head() -> ContractCase:
    in_sizes = [Size((1, 8, 4, 4)), Size((1, 16, 4, 4))]
    module = TransformerSegmentationHead(
        **_head_kwargs(in_sizes, original_in_shape=Size((3, 16, 16)))
    ).eval()
    return ContractCase(
        module=module,
        inputs={
            "x": [
                torch.randn(BATCH, size[1], size[2], size[3])
                for size in in_sizes
            ]
        },
        bindings={
            "B": BATCH,
            "N": len(in_sizes),
            "C": tuple(size[1] for size in in_sizes),
            "H": tuple(size[2] for size in in_sizes),
            "W": tuple(size[3] for size in in_sizes),
            "H_image": 16,
            "W_image": 16,
            "n_classes": N_CLASSES,
        },
    )


def _ddrnet_segmentation_head() -> list[ContractCase]:
    bindings: dict[str, int | tuple[int, ...]] = {
        "B": BATCH,
        "C_in": 128,
        "H": 4,
        "W": 4,
        "H_image": 8,
        "W_image": 8,
        "n_classes": N_CLASSES,
    }
    inputs = torch.randn(BATCH, 128, 4, 4)
    cases: list[ContractCase] = []
    for mode, inter_mode in [(None, "bilinear"), ("export", "pixel_shuffle")]:
        module = DDRNetSegmentationHead(
            inter_channels=8,
            inter_mode=inter_mode,
            **_head_kwargs(
                Size((128, 4, 4)), original_in_shape=Size((3, 8, 8))
            ),
        ).eval()
        module.export = mode == "export"
        cases.append(
            ContractCase(
                module=module,
                inputs={"inputs": inputs},
                bindings=dict(bindings),
                mode=mode,
            )
        )
    return cases


def _discsubnet_head() -> list[ContractCase]:
    bindings: dict[str, int | tuple[int, ...]] = {
        "B": BATCH,
        "C_in": 3,
        "H": 16,
        "W": 16,
        "n_classes": N_CLASSES,
    }
    inputs = {
        "reconstruction": torch.randn(BATCH, 3, 16, 16),
        "original": torch.randn(BATCH, 3, 16, 16),
    }
    cases: list[ContractCase] = []
    for mode in [None, "export"]:
        module = DiscSubNetHead(
            base_channels=4,
            width_multipliers=[1, 2],
            out_channels=N_CLASSES,
            **_head_kwargs(
                Size((3, 16, 16)), original_in_shape=Size((3, 16, 16))
            ),
        ).eval()
        module.export = mode == "export"
        cases.append(
            ContractCase(
                module=module,
                inputs=dict(inputs),
                bindings=dict(bindings),
                mode=mode,
            )
        )
    return cases


def _fomo_head() -> list[ContractCase]:
    torch.manual_seed(0)
    extra: dict[str, Any] = {"n_conv_layers": 2, "conv_channels": 8}
    kwargs = (
        _head_kwargs(Size((8, 4, 4)), original_in_shape=Size((3, 8, 8)))
        | extra
    )
    bindings: dict[str, int | tuple[int, ...]] = {
        "C_in": 8,
        "H": 4,
        "W": 4,
        "n_classes": N_CLASSES,
    }

    training = FOMOHead(**kwargs).train()
    export = FOMOHead(**kwargs).eval()
    export.export = True
    inference = FOMOHead(**kwargs).eval()

    batched = torch.randn(BATCH, 8, 4, 4)
    # The inference contract shares a single `K` across the batch, so
    # one image keeps the number of retained detections unambiguous.
    single = torch.randn(1, 8, 4, 4)
    return [
        ContractCase(
            module=training,
            inputs={"inputs": batched},
            bindings=bindings | {"B": BATCH},
            outputs=training(batched),
        ),
        ContractCase(
            module=export,
            inputs={"inputs": batched},
            bindings=bindings | {"B": BATCH},
            mode="export",
        ),
        ContractCase(
            module=inference,
            inputs={"inputs": single},
            bindings=bindings | {"B": 1},
            mode="inference",
        ),
    ]


def _ghostfacenet_head() -> ContractCase:
    module = GhostFaceNetHead(
        embedding_size=8,
        dropout=0.0,
        **_head_kwargs(Size((16, 2, 2)), original_in_shape=Size((3, 33, 35))),
    ).eval()
    return ContractCase(
        module=module,
        inputs={"x": torch.randn(BATCH, 16, 2, 2)},
        bindings={
            "B": BATCH,
            "C_in": 16,
            "H": 2,
            "W": 2,
            "embedding_size": 8,
        },
    )


def _ocr_ctc_head() -> list[ContractCase]:
    inputs = torch.randn(BATCH, 16, 1, 6)
    cases: list[ContractCase] = []
    for mid_channels in [None, 8]:
        module = OCRCTCHead(
            alphabet=list("abc"),
            mid_channels=mid_channels,
            **_head_kwargs(
                Size((16, 1, 6)), original_in_shape=Size((3, 32, 32))
            ),
        ).eval()
        cases.append(
            ContractCase(
                module=module,
                inputs={"x": inputs},
                bindings={
                    "B": BATCH,
                    "C_in": 16,
                    "W": 6,
                    "alphabet_size": module.out_channels,
                },
            )
        )
    return cases


def _detection_cases(
    head_type: type[BaseDetectionHead],
    *,
    kwargs: dict[str, Any] | None = None,
    training: dict[str, int] | None = None,
    export: dict[str, int] | None = None,
) -> list[ContractCase]:
    """Build the training and the export case of a detection head.

    The default `Outputs` group documents what the head returns while
    training, so the training case carries the outputs produced there
    rather than letting the runner re-run an evaluating module.
    """
    inputs = _detection_inputs()
    head_kwargs = _detection_kwargs() | (kwargs or {})
    training_head = head_type(**head_kwargs).train()
    export_head = head_type(**head_kwargs).eval()
    export_head.export = True
    return [
        ContractCase(
            module=training_head,
            inputs={"inputs": inputs},
            bindings=_detection_bindings(
                A=DETECTION_ANCHORS, **(training or {})
            ),
            outputs=training_head(inputs),
        ),
        ContractCase(
            module=export_head,
            inputs={"inputs": inputs},
            bindings=_detection_bindings(**(export or {})),
            mode="export",
        ),
    ]


CASES: dict[
    type[nn.Module], Callable[[], ContractCase | list[ContractCase]]
] = {
    BiSeNetHead: _bisenet_head,
    ClassificationHead: _classification_head,
    DDRNetSegmentationHead: _ddrnet_segmentation_head,
    DiscSubNetHead: _discsubnet_head,
    EfficientBBoxHead: lambda: _detection_cases(EfficientBBoxHead),
    EfficientKeypointBBoxHead: lambda: _detection_cases(
        EfficientKeypointBBoxHead,
        training={"n_keypoints": N_KEYPOINTS},
        export={"n_keypoints": N_KEYPOINTS},
    ),
    FOMOHead: _fomo_head,
    GhostFaceNetHead: _ghostfacenet_head,
    OCRCTCHead: _ocr_ctc_head,
    PrecisionBBoxHead: lambda: _detection_cases(
        PrecisionBBoxHead,
        kwargs={"reg_max": REG_MAX},
        training={"reg_max": REG_MAX},
    ),
    PrecisionSegmentBBoxHead: lambda: _detection_cases(
        PrecisionSegmentBBoxHead,
        kwargs={"n_masks": N_MASKS, "n_proto": 8, "reg_max": REG_MAX},
        training={"n_masks": N_MASKS, "reg_max": REG_MAX},
        export={"n_masks": N_MASKS},
    ),
    SegmentationHead: _segmentation_head,
    TransformerClassificationHead: _transformer_classification_head,
    TransformerSegmentationHead: _transformer_segmentation_head,
}

UNSUPPORTED: dict[type[nn.Module], str] = {
    BaseHead: "Abstract head, its forward contract is owned by BaseNode",
    BaseDetectionHead: (
        "Abstract detection head, its forward contract is owned by BaseNode"
    ),
}
