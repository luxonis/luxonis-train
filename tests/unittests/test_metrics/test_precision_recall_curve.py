from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.metrics import PrecisionRecallCurve
from luxonis_train.lightning.luxonis_lightning import LuxonisLightningModule
from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.heads.efficient_bbox_head import EfficientBBoxHead
from luxonis_train.nodes.heads.precision_bbox_head import PrecisionBBoxHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import non_max_suppression
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class DummyBBoxNode(BaseNode, register=False):
    task = Tasks.BOUNDINGBOX
    iou_thres = 0.45
    max_det = 300

    def forward(self, _: Tensor) -> Tensor:
        raise NotImplementedError


def make_node() -> DummyBBoxNode:
    return DummyBBoxNode(
        n_classes=2,
        original_in_shape=Size([3, 100, 100]),
        dataset_metadata=DatasetMetadata(
            classes={"": {"class0": 0, "class1": 1}}
        ),
    )


def make_metric(
    *,
    thresholds: list[float] | None = None,
) -> PrecisionRecallCurve:
    return PrecisionRecallCurve(
        node=make_node(),
        confidence_thresholds=thresholds or [0.0, 0.7, 0.85, 1.0],
        matching_iou_threshold=0.5,
        nms_iou_threshold=0.45,
        max_detections=300,
    )


def make_pre_nms(
    rows: list[tuple[float, float, float, float, float, int]],
    *,
    n_classes: int = 2,
) -> Tensor:
    output = torch.zeros(
        (1, len(rows), 5 + n_classes),
        dtype=torch.float32,
    )
    for index, (x1, y1, x2, y2, score, class_id) in enumerate(rows):
        output[0, index, :4] = torch.tensor([x1, y1, x2, y2])
        output[0, index, 4] = 1.0
        output[0, index, 5 + class_id] = score
    return output


def test_precision_recall_curve_uses_node_nms_defaults() -> None:
    metric = PrecisionRecallCurve(
        node=make_node(),
        confidence_thresholds=[0.0, 0.5, 1.0],
    )

    assert metric.nms_iou_threshold == 0.45
    assert metric.max_detections == 300


def test_precision_recall_curve_values() -> None:
    metric = make_metric()
    predictions = make_pre_nms(
        [
            (0, 0, 20, 10, 0.9, 0),
            (0, 10, 20, 20, 0.8, 0),
            (50, 50, 70, 70, 0.6, 1),
        ]
    )
    targets = torch.tensor(
        [[0, 0, 0.0, 0.0, 0.2, 0.2]],
        dtype=torch.float32,
    )

    metric.update(predictions, targets)
    result = metric.compute()

    torch.testing.assert_close(
        result["confidence"],
        torch.tensor([0.0, 0.7, 0.85, 1.0]),
    )
    torch.testing.assert_close(
        result["precision"],
        torch.tensor([1 / 3, 1 / 2, 1.0, 0.0]),
    )
    torch.testing.assert_close(
        result["recall"],
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    )
    torch.testing.assert_close(
        result["f1"],
        torch.tensor([0.5, 2 / 3, 1.0, 0.0]),
    )


def test_precision_recall_curve_accumulates_and_resets() -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    predictions = make_pre_nms([(10, 10, 30, 30, 0.9, 0)])
    targets = torch.tensor(
        [[0, 0, 0.1, 0.1, 0.2, 0.2]],
        dtype=torch.float32,
    )

    metric.update(predictions, targets)
    metric.update(predictions, targets)

    torch.testing.assert_close(
        metric.true_positives,
        torch.tensor([2, 2, 0]),
    )
    torch.testing.assert_close(
        metric.false_positives,
        torch.tensor([0, 0, 0]),
    )
    assert int(metric.target_count) == 2

    metric.reset()

    torch.testing.assert_close(
        metric.true_positives,
        torch.zeros(3, dtype=torch.long),
    )
    torch.testing.assert_close(
        metric.false_positives,
        torch.zeros(3, dtype=torch.long),
    )
    assert int(metric.target_count) == 0


@pytest.mark.parametrize(
    ("predictions", "targets"),
    [
        (
            torch.empty((1, 0, 7)),
            torch.tensor([[0, 0, 0.1, 0.1, 0.2, 0.2]]),
        ),
        (
            make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
            torch.empty((0, 6)),
        ),
        (
            torch.empty((1, 0, 7)),
            torch.empty((0, 6)),
        ),
    ],
)
def test_precision_recall_curve_empty_inputs(
    predictions: Tensor,
    targets: Tensor,
) -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])

    metric.update(predictions.float(), targets.float())
    result = metric.compute()

    assert torch.isfinite(result["precision"]).all()
    assert torch.isfinite(result["recall"]).all()
    assert torch.isfinite(result["f1"]).all()


def test_precision_recall_curve_run_update_routing() -> None:
    metric = make_metric()
    predictions = make_pre_nms([(10, 10, 30, 30, 0.9, 0)])
    targets = torch.tensor(
        [[0, 0, 0.1, 0.1, 0.2, 0.2]],
        dtype=torch.float32,
    )

    parameters = metric.get_parameters(
        {
            "boundingbox": [torch.empty((0, 6))],
            "detections_pre_nms": predictions,
        },
        {"/boundingbox": targets},
    )

    assert set(parameters) == {
        "detections_pre_nms",
        "target_boundingbox",
    }

    metric.run_update(
        {
            "boundingbox": [torch.empty((0, 6))],
            "detections_pre_nms": predictions,
        },
        {"/boundingbox": targets},
    )
    assert int(metric.target_count) == 1


def test_precision_recall_curve_runs_nms_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxonis_train.attached_modules.metrics.precision_recall_curve as module

    calls = 0
    original = module.non_max_suppression

    def counted_nms(*args, **kwargs) -> list[Tensor]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "non_max_suppression", counted_nms)

    metric = make_metric()
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )

    assert calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_thresholds": [0.5]},
        {"confidence_thresholds": [0.5, 0.5]},
        {"confidence_thresholds": [-0.1, 0.5]},
        {"num_thresholds": 1},
        {"min_confidence": 0.8, "max_confidence": 0.2},
        {"matching_iou_threshold": 1.1},
        {"nms_iou_threshold": -0.1},
        {"max_detections": 0},
    ],
)
def test_precision_recall_curve_rejects_invalid_configuration(
    kwargs: dict[str, Any],
) -> None:
    params: dict[str, Any] = {
        "node": make_node(),
        "nms_iou_threshold": 0.45,
        "max_detections": 300,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match="must"):
        PrecisionRecallCurve(**params)


class DummyNodes(dict[str, object]):
    def formatted_name(self, name: str) -> str:
        return name


class DummyTracker:
    def __init__(self) -> None:
        self.images: list[dict[str, object]] = []

    def log_image(
        self,
        name: str,
        img: object,
        step: int,
    ) -> None:
        self.images.append({"name": name, "img": img, "step": step})

    def log_matrix(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Matrix logging should not be used for curves.")


def make_epoch_end_harness(
    metric: PrecisionRecallCurve,
    *,
    is_global_zero: bool,
    log_sub_metrics: bool = True,
) -> SimpleNamespace:
    tracker = DummyTracker()
    logged: list[tuple[str, Tensor, bool]] = []

    def log(name: str, value: Tensor, sync_dist: bool) -> None:
        logged.append((name, value, sync_dist))

    def print_results(**kwargs: object) -> None:
        del kwargs

    node = SimpleNamespace(
        metrics={"PrecisionRecallCurve": metric},
        losses={},
        visualizers={},
    )
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            log_sub_metrics=log_sub_metrics,
            n_log_images=0,
            validation_interval=1,
            epochs=2,
            run_validation_after_first_epoch=False,
            callbacks=[],
        ),
        exporter=SimpleNamespace(name=None),
        model=SimpleNamespace(name="dummy"),
    )

    return SimpleNamespace(
        _loss_accumulators={"val": {"loss": torch.tensor(0.0)}},
        nodes=DummyNodes({"head": node}),
        cfg=cfg,
        trainer=SimpleNamespace(
            strategy=object(),
            is_global_zero=is_global_zero,
        ),
        device=torch.device("cpu"),
        progress_bar=SimpleNamespace(),
        tracker=tracker,
        current_epoch=3,
        log=log,
        _print_results=print_results,
        _n_logged_images=0,
        _sequentially_logged_visualizations=[],
        _needs_vis_buffering=True,
        _class_log_counts=[],
        _logged=logged,
    )


def test_precision_recall_curve_separates_scalars_and_artifact() -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )

    computed = metric.compute()
    primary, submetrics = metric.get_loggable_values(computed)
    artifacts = metric.get_artifacts(computed)

    assert set(computed) == {
        "confidence",
        "precision",
        "recall",
        "f1",
        "max_f1",
        "confidence_at_max_f1",
    }
    torch.testing.assert_close(primary, computed["max_f1"])
    assert set(submetrics) == {"confidence_at_max_f1"}
    assert all(value.ndim == 0 for value in submetrics.values())
    assert metric.get_artifact_names() == ("curves",)
    assert set(artifacts) == {"curves"}
    assert artifacts["curves"].ndim == 3
    assert artifacts["curves"].shape[0] == 3
    assert artifacts["curves"].dtype == torch.uint8


def test_evaluation_epoch_end_logs_curve_once_and_resets() -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )
    harness = make_epoch_end_harness(metric, is_global_zero=True)

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )

    metric_logs = [
        name
        for name, _, _ in harness._logged
        if name.startswith("val/metric/")
    ]
    assert metric_logs == [
        "val/metric/head/PrecisionRecallCurve",
        "val/metric/head/confidence_at_max_f1",
    ]
    assert len(harness.tracker.images) == 1
    assert harness.tracker.images[0]["name"] == (
        "val/metrics/3/head/PrecisionRecallCurve/curves"
    )
    assert int(metric.target_count) == 0
    torch.testing.assert_close(
        metric.true_positives,
        torch.zeros(3, dtype=torch.long),
    )


def test_evaluation_epoch_end_logs_primary_without_submetrics() -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )
    harness = make_epoch_end_harness(
        metric,
        is_global_zero=True,
        log_sub_metrics=False,
    )

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )

    metric_logs = [
        name
        for name, _, _ in harness._logged
        if name.startswith("val/metric/")
    ]
    assert metric_logs == ["val/metric/head/PrecisionRecallCurve"]
    assert len(harness.tracker.images) == 1
    assert int(metric.target_count) == 0


def test_evaluation_epoch_end_skips_artifact_on_nonzero_rank() -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )
    harness = make_epoch_end_harness(metric, is_global_zero=False)

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )

    assert harness.tracker.images == []
    assert int(metric.target_count) == 0


def test_evaluation_epoch_end_skips_invalid_artifact_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )

    def invalid_artifacts(
        values: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        del values
        return {"curves": torch.zeros((10, 10))}

    monkeypatch.setattr(metric, "get_artifacts", invalid_artifacts)
    harness = make_epoch_end_harness(metric, is_global_zero=True)

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )

    metric_logs = [
        name
        for name, _, _ in harness._logged
        if name.startswith("val/metric/")
    ]
    assert metric_logs == [
        "val/metric/head/PrecisionRecallCurve",
        "val/metric/head/confidence_at_max_f1",
    ]
    assert harness.tracker.images == []
    assert int(metric.target_count) == 0


def test_evaluation_epoch_end_skips_artifact_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )

    def failing_artifacts(
        values: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        del values
        raise RuntimeError("artifact rendering failed")

    monkeypatch.setattr(metric, "get_artifacts", failing_artifacts)
    harness = make_epoch_end_harness(metric, is_global_zero=True)

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )

    metric_logs = [
        name
        for name, _, _ in harness._logged
        if name.startswith("val/metric/")
    ]
    assert metric_logs == [
        "val/metric/head/PrecisionRecallCurve",
        "val/metric/head/confidence_at_max_f1",
    ]
    assert harness.tracker.images == []
    assert int(metric.target_count) == 0


def test_evaluation_epoch_end_skips_tracker_image_logging_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )
    harness = make_epoch_end_harness(metric, is_global_zero=True)

    def failing_log_image(
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        raise RuntimeError("tracker image logging failed")

    monkeypatch.setattr(
        harness.tracker,
        "log_image",
        failing_log_image,
    )

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness),
        "val",
    )

    metric_logs = [
        name
        for name, _, _ in harness._logged
        if name.startswith("val/metric/")
    ]
    assert metric_logs == [
        "val/metric/head/PrecisionRecallCurve",
        "val/metric/head/confidence_at_max_f1",
    ]
    assert harness.tracker.images == []
    assert int(metric.target_count) == 0


def test_mlflow_keys_classify_curve_as_image_artifact() -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    harness = make_epoch_end_harness(metric, is_global_zero=True)

    keys = LuxonisLightningModule.get_mlflow_logging_keys(
        cast(LuxonisLightningModule, harness)
    )

    assert "val/metric/head/PrecisionRecallCurve" in keys["metrics"]
    assert "val/metric/head/confidence_at_max_f1" in keys["metrics"]
    assert (
        "val/metrics/0/head/PrecisionRecallCurve/curves.png"
        in keys["artifacts"]
    )
    assert (
        "test/metrics/2/head/PrecisionRecallCurve/curves.png"
        in keys["artifacts"]
    )
    assert not any(
        name.endswith(("/precision", "/recall", "/confidence", "/f1"))
        for name in keys["metrics"]
    )


_REAL_BATCH_SIZE = 2
_REAL_IMAGE_HEIGHT = 64
_REAL_IMAGE_WIDTH = 64
_REAL_N_CLASSES = 2
_REAL_FEATURE_SHAPES = (
    Size([_REAL_BATCH_SIZE, 16, 8, 8]),
    Size([_REAL_BATCH_SIZE, 32, 4, 4]),
    Size([_REAL_BATCH_SIZE, 64, 2, 2]),
)


class _RealEfficientBBoxHead(EfficientBBoxHead, register=False):
    task = Tasks.BOUNDINGBOX
    original_in_shape = cast(
        Any, Size([3, _REAL_IMAGE_HEIGHT, _REAL_IMAGE_WIDTH])
    )
    n_classes = cast(Any, _REAL_N_CLASSES)

    @property
    def input_shapes(self) -> list[Packet[Size]]:
        return [{"features": list(_REAL_FEATURE_SHAPES)}]

    @property
    def in_sizes(self) -> list[Size]:
        return list(_REAL_FEATURE_SHAPES)


class _RealPrecisionBBoxHead(PrecisionBBoxHead, register=False):
    task = Tasks.BOUNDINGBOX
    original_in_shape = cast(
        Any, Size([3, _REAL_IMAGE_HEIGHT, _REAL_IMAGE_WIDTH])
    )
    n_classes = cast(Any, _REAL_N_CLASSES)

    @property
    def input_shapes(self) -> list[Packet[Size]]:
        return [{"features": list(_REAL_FEATURE_SHAPES)}]

    @property
    def in_sizes(self) -> list[Size]:
        return list(_REAL_FEATURE_SHAPES)


def _make_real_features(seed: int) -> list[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(shape, generator=generator)
        for shape in _REAL_FEATURE_SHAPES
    ]


@pytest.mark.parametrize(
    ("head_cls", "seed"),
    [
        (_RealEfficientBBoxHead, 1301),
        (_RealPrecisionBBoxHead, 1302),
    ],
)
def test_real_bbox_head_to_precision_recall_curve_e2e(
    head_cls: type[EfficientBBoxHead | PrecisionBBoxHead],
    seed: int,
) -> None:
    torch.manual_seed(seed)

    head = head_cls(
        n_heads=3,
        conf_thres=0.0,
        iou_thres=0.45,
        max_det=50,
        dataset_metadata=DatasetMetadata(
            classes={"": {"class0": 0, "class1": 1}}
        ),
        task_name="",
    )
    head.eval()

    with torch.inference_mode():
        packet = head(_make_real_features(seed))

    assert {"boundingbox", "detections_pre_nms"} <= set(packet)

    detections_pre_nms = packet["detections_pre_nms"]
    boundingbox = packet["boundingbox"]

    assert isinstance(detections_pre_nms, Tensor)
    assert isinstance(boundingbox, list)
    assert detections_pre_nms.shape == (
        _REAL_BATCH_SIZE,
        sum(shape[-2] * shape[-1] for shape in _REAL_FEATURE_SHAPES),
        5 + _REAL_N_CLASSES,
    )
    assert len(boundingbox) == _REAL_BATCH_SIZE

    replay = non_max_suppression(
        detections_pre_nms,
        n_classes=head.n_classes,
        conf_thres=head.conf_thres,
        iou_thres=head.iou_thres,
        bbox_format="xyxy",
        max_det=head.max_det,
        predicts_objectness=False,
    )
    for actual, expected in zip(boundingbox, replay, strict=True):
        torch.testing.assert_close(actual, expected)

    image_index, detection = next(
        (index, detections[0])
        for index, detections in enumerate(boundingbox)
        if detections.numel() > 0
    )
    x1, y1, x2, y2 = detection[:4]
    score = float(detection[4])
    class_id = float(detection[5])

    target = detection.new_tensor(
        [
            [
                float(image_index),
                class_id,
                float(x1) / _REAL_IMAGE_WIDTH,
                float(y1) / _REAL_IMAGE_HEIGHT,
                float(x2 - x1) / _REAL_IMAGE_WIDTH,
                float(y2 - y1) / _REAL_IMAGE_HEIGHT,
            ]
        ]
    )
    interior_score = min(max(score, 1e-6), 1.0 - 1e-6)

    metric = PrecisionRecallCurve(
        node=head,
        confidence_thresholds=[0.0, interior_score, 1.0],
        matching_iou_threshold=0.5,
    )
    labels = {f"{head.task_name}/boundingbox": target}

    parameters = metric.get_parameters(packet, labels)
    assert set(parameters) == {
        "detections_pre_nms",
        "target_boundingbox",
    }

    metric.run_update(packet, labels)
    computed = metric.compute()
    primary, submetrics = metric.get_loggable_values(computed)
    artifacts = metric.get_artifacts(computed)

    assert float(computed["max_f1"]) > 0
    torch.testing.assert_close(primary, computed["max_f1"])
    assert set(submetrics) == {"confidence_at_max_f1"}
    assert all(value.numel() == 1 for value in submetrics.values())

    artifact = artifacts["curves"]
    assert artifact.ndim == 3
    assert artifact.shape[0] == 3
    assert artifact.dtype == torch.uint8
