from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from torch import Size, Tensor

from luxonis_train.nodes import BaseNode
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
from luxonis_train.nodes.heads.precision_seg_bbox_head import (
    refine_and_apply_masks,
)
from luxonis_train.utils import DatasetMetadata

from .shape_contracts import assert_contract

ORIGINAL_IN_SHAPE = Size((3, 32, 32))
DETECTION_SIZES = [Size((8, 8, 8)), Size((8, 4, 4))]
DETECTION_ANCHORS = 8 * 8 + 4 * 4


def _metadata(n_classes: int = 2, n_keypoints: int = 2) -> DatasetMetadata:
    return DatasetMetadata(
        classes={"": {f"class-{i}": i for i in range(n_classes)}},
        n_keypoints={"": n_keypoints},
    )


def _head_kwargs(
    in_sizes: Size | list[Size],
    *,
    n_classes: int = 2,
    n_keypoints: int = 2,
    original_in_shape: Size = ORIGINAL_IN_SHAPE,
) -> dict[str, Any]:
    return {
        "in_sizes": in_sizes,
        "original_in_shape": original_in_shape,
        "n_classes": n_classes,
        "n_keypoints": n_keypoints,
        "dataset_metadata": _metadata(n_classes, n_keypoints),
    }


def test_classification_head_and_base_head(monkeypatch: pytest.MonkeyPatch):
    node = ClassificationHead(
        dropout_rate=0.0,
        **_head_kwargs(Size((8, 4, 4))),
    ).eval()

    inputs = torch.randn(2, 8, 4, 4)
    assert_contract(
        ClassificationHead,
        {"inputs": inputs},
        {"logits": node(inputs)},
        {"B": 2, "C_in": 8, "H": 4, "W": 4, "n_classes": 2},
    )
    assert node.get_head_config() == {
        "parser": "ClassificationParser",
        "metadata": {
            "classes": ["class-0", "class-1"],
            "n_classes": 2,
            "is_softmax": False,
        },
    }

    expected = iter([{"file": "image.jpg"}])
    annotate = Mock(return_value=expected)
    monkeypatch.setattr(
        "luxonis_train.nodes.heads.base_head.default_annotate", annotate
    )
    result = node.annotate(
        {"classification": torch.randn(1, 2)},
        [Path("image.jpg")],
        Mock(),
    )
    assert result is expected
    annotate.assert_called_once()


def test_segmentation_heads():
    kwargs = _head_kwargs(
        Size((16, 8, 8)), original_in_shape=Size((3, 32, 32))
    )
    inputs = torch.randn(2, 16, 8, 8)
    bindings = {
        "B": 2,
        "C_in": 16,
        "H": 8,
        "W": 8,
        "H_image": 32,
        "W_image": 32,
        "n_classes": 2,
    }

    segmentation = SegmentationHead(**kwargs).eval()
    assert_contract(
        SegmentationHead,
        {"inputs": inputs},
        {"logits": segmentation(inputs)},
        bindings,
    )
    assert segmentation.get_custom_head_config() == {"is_softmax": False}

    bisenet = BiSeNetHead(intermediate_channels=8, **kwargs).eval()
    assert_contract(
        BiSeNetHead,
        {"inputs": inputs},
        {"logits": bisenet(inputs)},
        bindings,
    )
    assert bisenet.get_custom_head_config() == {"is_softmax": False}


def test_transformer_heads():
    classification = TransformerClassificationHead(
        dropout_rate=0.0,
        **_head_kwargs(Size((8,))),
    ).eval()
    x = torch.randn(2, 8)
    assert_contract(
        TransformerClassificationHead,
        {"x": x},
        {"logits": classification(x)},
        {"B": 2, "C_in": 8, "n_classes": 2},
    )
    assert classification.get_custom_head_config() == {"is_softmax": False}
    assert BaseHead.get_custom_head_config(classification) == {}
    classification._in_sizes = [Size((8,)), Size((8,))]
    with pytest.raises(TypeError, match="single"):
        _ = classification.in_channels

    segmentation = TransformerSegmentationHead(
        **_head_kwargs(
            [Size((1, 8, 4, 4)), Size((1, 16, 4, 4))],
            original_in_shape=Size((3, 16, 16)),
        )
    ).eval()
    features = [torch.randn(2, 8, 4, 4), torch.randn(2, 16, 4, 4)]
    assert_contract(
        TransformerSegmentationHead,
        {"x": features},
        {"logits": segmentation(features)},
        {
            "B": 2,
            "N": 2,
            "C": (8, 16),
            "H": (4, 4),
            "W": (4, 4),
            "H_image": 16,
            "W_image": 16,
            "n_classes": 2,
        },
    )


def test_ddrnet_segmentation_head_modes():
    kwargs = _head_kwargs(Size((128, 4, 4)), original_in_shape=Size((3, 8, 8)))
    inputs = torch.randn(2, 128, 4, 4)
    bindings = {
        "B": 2,
        "C_in": 128,
        "H": 4,
        "W": 4,
        "H_image": 8,
        "W_image": 8,
        "n_classes": 2,
    }
    head = DDRNetSegmentationHead(
        inter_channels=8,
        inter_mode="pixel_shuffle",
        **kwargs,
    ).eval()
    assert_contract(
        DDRNetSegmentationHead,
        {"inputs": inputs},
        {"logits": head(inputs)},
        bindings,
    )
    assert head.get_weights_url().endswith("ddrnet_head_23slim_coco.ckpt")
    assert head.get_custom_head_config() == {"is_softmax": False}

    head.export = True
    assert_contract(
        DDRNetSegmentationHead,
        {"inputs": inputs},
        {"labels": head(inputs)},
        bindings,
        mode="export",
    )
    assert head(inputs).dtype == torch.int32

    binary = DDRNetSegmentationHead(
        inter_channels=4,
        inter_mode="bilinear",
        **_head_kwargs(
            Size((256, 4, 4)),
            n_classes=1,
            original_in_shape=Size((3, 8, 8)),
        ),
    ).eval()
    assert binary.get_weights_url().endswith("ddrnet_head_23_coco.ckpt")
    binary.export = True
    assert binary(torch.randn(2, 256, 4, 4)).dtype == torch.int32

    with pytest.raises(ValueError, match="multiple"):
        DDRNetSegmentationHead(
            inter_channels=6,
            inter_mode="pixel_shuffle",
            **kwargs,
        )

    unsupported = DDRNetSegmentationHead(
        inter_channels=4,
        **_head_kwargs(Size((16, 4, 4)), original_in_shape=Size((3, 8, 8))),
    )
    with pytest.raises(NotImplementedError, match="No online weights"):
        unsupported.get_weights_url()

    with patch.object(BaseNode, "load_checkpoint") as load:
        unsupported.load_checkpoint("checkpoint.ckpt")
    load.assert_called_once_with("checkpoint.ckpt", strict=False)


def test_discsubnet_head():
    node = DiscSubNetHead(
        base_channels=4,
        width_multipliers=[1, 2],
        out_channels=2,
        **_head_kwargs(Size((3, 16, 16))),
    ).eval()
    reconstruction = torch.randn(2, 3, 16, 16)
    original = torch.randn(2, 3, 16, 16)

    bindings = {"B": 2, "C_in": 3, "H": 16, "W": 16, "n_classes": 2}
    inputs = {"reconstruction": reconstruction, "original": original}

    output = node(**inputs)
    assert_contract(DiscSubNetHead, inputs, output, bindings)
    assert output["reconstruction"] is reconstruction

    node.export = True
    assert_contract(
        DiscSubNetHead,
        inputs,
        node(**inputs),
        bindings,
        mode="export",
    )
    assert DiscSubNetHead.get_variants()[0] == "n"


@pytest.mark.parametrize("use_nms", [False, True])
def test_fomo_head_modes(use_nms: bool):
    node = FOMOHead(
        n_conv_layers=2,
        conv_channels=8,
        use_nms=use_nms,
        **_head_kwargs(Size((8, 4, 4)), original_in_shape=Size((3, 8, 8))),
    )
    inputs = torch.randn(2, 8, 4, 4)
    bindings = {"B": 2, "C_in": 8, "H": 4, "W": 4, "n_classes": 2}

    assert_contract(FOMOHead, {"inputs": inputs}, node(inputs), bindings)

    node.eval()
    output = node(inputs)
    # The inference contract shares a single number of detections
    # across the batch, which random logits do not guarantee, so the
    # structure is checked by hand here.
    assert set(output) == {"keypoints", "heatmap"}
    assert len(output["keypoints"]) == 2
    assert node.n_keypoints == 1

    empty = node._heatmap_to_kpts(torch.full((1, 2, 2, 2), -100.0))
    assert empty[0].shape == (0, 1, 4)

    node.export = True
    assert_contract(
        FOMOHead,
        {"inputs": inputs},
        node(inputs),
        bindings,
        mode="export",
    )


def test_ghostfacenet_head():
    node = GhostFaceNetHead(
        embedding_size=8,
        dropout=0.0,
        **_head_kwargs(Size((16, 2, 2)), original_in_shape=Size((3, 33, 35))),
    ).eval()
    node.initialize_weights("none")
    x = torch.randn(2, 16, 2, 2)
    assert_contract(
        GhostFaceNetHead,
        {"x": x},
        {"embedding": node(x)},
        {"B": 2, "C_in": 16, "H": 2, "W": 2, "embedding_size": 8},
    )


@pytest.mark.parametrize("mid_channels", [None, 8])
def test_ocr_ctc_head(mid_channels: int | None):
    node = OCRCTCHead(
        alphabet=list("abc"),
        mid_channels=mid_channels,
        return_feats=True,
        **_head_kwargs(Size((16, 1, 6))),
    ).eval()
    node.initialize_weights("none")
    inputs = torch.randn(2, 16, 1, 6)

    assert_contract(
        OCRCTCHead,
        {"x": inputs},
        {"logits": node(inputs)},
        {
            "B": 2,
            "C_in": 16,
            "W": 6,
            "alphabet_size": node.out_channels,
        },
    )
    assert node.encoder.alphabet[0] == ""
    assert node.encoder.alphabet[1:] == list("abc")
    assert node.decoder is not None
    assert node.export_output_names == ["output_ocr_ctc"]
    assert node.get_custom_head_config()["is_softmax"] is True

    node.export = True
    torch.testing.assert_close(
        node(inputs).sum(dim=-1), torch.ones(2, 6), atol=1e-6, rtol=1e-6
    )


def _detection_kwargs() -> dict[str, Any]:
    return _head_kwargs(
        DETECTION_SIZES,
        n_classes=2,
        n_keypoints=2,
        original_in_shape=Size((3, 64, 64)),
    )


def _detection_inputs() -> list[Tensor]:
    return [torch.randn(2, *size) for size in DETECTION_SIZES]


def _detection_bindings(**extra: int) -> dict[str, Any]:
    return {
        "B": 2,
        "N": len(DETECTION_SIZES),
        "C": tuple(size[0] for size in DETECTION_SIZES),
        "H": tuple(size[1] for size in DETECTION_SIZES),
        "W": tuple(size[2] for size in DETECTION_SIZES),
        "n_classes": 2,
        **extra,
    }


def test_efficient_bbox_head_modes():
    node = EfficientBBoxHead(
        n_heads=3,
        conf_thres=0.99,
        iou_thres=0.5,
        max_det=10,
        **_detection_kwargs(),
    )
    inputs = _detection_inputs()
    assert node.n_heads == 2
    assert node.stride.tolist() == [8, 16]
    assert_contract(
        EfficientBBoxHead,
        {"inputs": inputs},
        node(inputs),
        _detection_bindings(A=DETECTION_ANCHORS),
    )

    node.initialize_weights("none")
    node.eval()
    node.request_detections_pre_nms()
    # The contract documents the training outputs, the decoded and the
    # pre-NMS detections are deliberately left undocumented.
    assert set(node(inputs)) == {
        "boundingbox",
        "features",
        "class_scores",
        "distributions",
        "detections_pre_nms",
    }
    assert node.keep_detections_pre_nms

    node.export = True
    assert_contract(
        EfficientBBoxHead,
        {"inputs": inputs},
        node(inputs),
        _detection_bindings(),
        mode="export",
    )
    assert node.export_output_names == ["output1_yolov6r2", "output2_yolov6r2"]
    assert node.get_custom_head_config()["subtype"] == "yolov6r2"
    node._variant = "n"
    assert node.get_weights_url().endswith("efficientbbox_head_n_coco.ckpt")

    with patch.object(BaseNode, "load_checkpoint") as load:
        node.load_checkpoint("checkpoint.ckpt")
    load.assert_called_once_with("checkpoint.ckpt", strict=False)


def test_detection_output_names():
    correct = EfficientBBoxHead(
        n_heads=2,
        export_output_names=["small", "large"],
        **_detection_kwargs(),
    )
    assert correct.export_output_names == ["small", "large"]

    wrong = EfficientBBoxHead(
        n_heads=2,
        export_output_names=["only-one"],
        **_detection_kwargs(),
    )
    assert wrong.export_output_names == [
        "output1_yolov6r2",
        "output2_yolov6r2",
    ]


def test_efficient_keypoint_bbox_head_modes():
    node = EfficientKeypointBBoxHead(
        n_heads=2,
        conf_thres=0.99,
        max_det=10,
        **_detection_kwargs(),
    )
    inputs = _detection_inputs()
    assert_contract(
        EfficientKeypointBBoxHead,
        {"inputs": inputs},
        node(inputs),
        _detection_bindings(A=DETECTION_ANCHORS, n_keypoints=2),
    )

    node.eval()
    node.request_detections_pre_nms()
    output = node(inputs)
    assert "detections_pre_nms" in output
    assert len(output["keypoints"]) == 2
    assert node.get_custom_head_config()["n_keypoints"] == 2
    assert node.export_output_names == [
        "output1_yolov6",
        "output2_yolov6",
        "kpt_output1",
        "kpt_output2",
    ]

    node.export = True
    assert_contract(
        EfficientKeypointBBoxHead,
        {"inputs": inputs},
        node(inputs),
        _detection_bindings(n_keypoints=2),
        mode="export",
    )


@pytest.mark.parametrize("reg_max", [1, 4])
def test_precision_bbox_head_modes(reg_max: int):
    node = PrecisionBBoxHead(
        n_heads=2,
        reg_max=reg_max,
        conf_thres=0.99,
        max_det=10,
        **_detection_kwargs(),
    )
    inputs = _detection_inputs()
    assert_contract(
        PrecisionBBoxHead,
        {"inputs": inputs},
        node(inputs),
        _detection_bindings(reg_max=reg_max),
    )
    node.initialize_weights("none")

    node.eval()
    node.request_detections_pre_nms()
    assert set(node(inputs)) == {
        "features",
        "boundingbox",
        "detections_pre_nms",
    }
    assert node.get_custom_head_config()["subtype"] == "yolov8"

    node.export = True
    assert_contract(
        PrecisionBBoxHead,
        {"inputs": inputs},
        node(inputs),
        _detection_bindings(),
        mode="export",
    )
    assert node.export_output_names == ["output1_yolov8", "output2_yolov8"]


def test_precision_segment_bbox_head_modes():
    node = PrecisionSegmentBBoxHead(
        n_heads=2,
        n_masks=4,
        n_proto=8,
        reg_max=4,
        conf_thres=0.99,
        max_det=10,
        **_detection_kwargs(),
    )
    inputs = _detection_inputs()
    assert_contract(
        PrecisionSegmentBBoxHead,
        {"inputs": inputs},
        node(inputs),
        _detection_bindings(A=DETECTION_ANCHORS, n_masks=4, reg_max=4),
    )

    node.eval()
    node.request_detections_pre_nms()
    output = node(inputs)
    assert "detections_pre_nms" in output
    assert len(output["boundingbox"]) == 2
    assert len(output["instance_segmentation"]) == 2

    node.export = True
    assert_contract(
        PrecisionSegmentBBoxHead,
        {"inputs": inputs},
        node(inputs),
        _detection_bindings(n_masks=4),
        mode="export",
    )
    assert node.export_output_names == [
        "output1_yolov8",
        "output2_yolov8",
        "output1_masks",
        "output2_masks",
        "protos_output",
    ]


def test_refine_and_apply_masks():
    empty = refine_and_apply_masks(
        torch.randn(4, 8, 8),
        torch.empty(0, 4),
        torch.empty(0, 4),
        height=16,
        width=16,
    )
    assert empty.shape == (0, 16, 16)
    assert empty.dtype == torch.uint8

    masks = refine_and_apply_masks(
        torch.randn(4, 8, 8),
        torch.randn(2, 4),
        torch.tensor([[0.0, 0.0, 8.0, 8.0], [4.0, 4.0, 16.0, 16.0]]),
        height=16,
        width=16,
        upsample=True,
    )
    assert masks.shape == (2, 16, 16)
