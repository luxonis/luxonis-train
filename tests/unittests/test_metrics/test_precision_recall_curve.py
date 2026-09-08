from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.metrics import PrecisionRecallCurve
from luxonis_train.attached_modules.metrics.precision_recall_curve import (
    _exclusive_threshold,
)
from luxonis_train.lightning.luxonis_lightning import LuxonisLightningModule
from luxonis_train.lightning.utils import (
    log_metric_artifacts,
    metric_artifact_image_name,
    mlflow_image_key,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.heads.base_detection_head import BaseDetectionHead
from luxonis_train.nodes.heads.efficient_bbox_head import EfficientBBoxHead
from luxonis_train.nodes.heads.efficient_keypoint_bbox_head import (
    EfficientKeypointBBoxHead,
)
from luxonis_train.nodes.heads.precision_bbox_head import PrecisionBBoxHead
from luxonis_train.nodes.heads.precision_seg_bbox_head import (
    PrecisionSegmentBBoxHead,
)
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import IncompatibleError, non_max_suppression
from luxonis_train.utils.dataset_metadata import DatasetMetadata

_FEATURE_SHAPES = (
    Size([1, 16, 8, 8]),
    Size([1, 32, 4, 4]),
    Size([1, 64, 2, 2]),
)


class DummyBBoxHead(BaseDetectionHead, register=False):
    """Detection head that only provides the NMS parameters the metric
    reads off its node.
    """

    task = Tasks.BOUNDINGBOX

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        raise NotImplementedError


class DummyBBoxNode(BaseNode, register=False):
    """Bounding box node that is not a detection head."""

    task = Tasks.BOUNDINGBOX

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        raise NotImplementedError


def make_node() -> DummyBBoxHead:
    return DummyBBoxHead(
        n_heads=3,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        n_classes=2,
        original_in_shape=Size([3, 100, 100]),
        in_sizes=list(_FEATURE_SHAPES),
        dataset_metadata=DatasetMetadata(
            classes={"": {"class0": 0, "class1": 1}}
        ),
    )


def make_metric(
    *,
    thresholds: list[float] | None = None,
    **kwargs: Any,
) -> PrecisionRecallCurve:
    return PrecisionRecallCurve(
        node=make_node(),
        confidence_thresholds=thresholds or [0.0, 0.7, 0.85, 1.0],
        matching_iou_threshold=0.5,
        nms_iou_threshold=0.45,
        max_detections=300,
        **kwargs,
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


def test_precision_recall_curve_rejects_non_detection_head() -> None:
    """The metric reads C{iou_thres}/C{max_det} off its node, which only
    detection heads define.

    Without the C{node} annotation this failed with a bare
    C{AttributeError} from C{nn.Module.__getattr__} instead of the
    framework's actionable error.
    """
    node = DummyBBoxNode(
        n_classes=2,
        original_in_shape=Size([3, 100, 100]),
        dataset_metadata=DatasetMetadata(
            classes={"": {"class0": 0, "class1": 1}}
        ),
    )

    with pytest.raises(IncompatibleError, match="BaseDetectionHead"):
        PrecisionRecallCurve(node=node, confidence_thresholds=[0.0, 0.5, 1.0])


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
        torch.tensor([1 / 3, 1 / 2, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        result["recall"],
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    )
    torch.testing.assert_close(
        result["f1"],
        torch.tensor([0.5, 2 / 3, 1.0, 0.0]),
    )


def test_precision_is_one_where_nothing_is_predicted() -> None:
    """Precision is undefined without predictions and is reported as
    C{1}, the value the Precision-Confidence curve converges to.

    Reporting C{0} instead dragged the curve to the floor over the whole
    confidence range above the highest prediction score, and added a
    spurious C{(0, 0)} endpoint to the PR curve.
    """
    metric = make_metric(thresholds=[0.0, 0.5, 0.95])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor([[0, 0, 0.1, 0.1, 0.2, 0.2]], dtype=torch.float32),
    )

    result = metric.compute()

    assert int(metric.true_positives[-1]) == 0
    assert int(metric.false_positives[-1]) == 0
    assert float(result["precision"][-1]) == 1.0
    assert float(result["recall"][-1]) == 0.0
    # A precision of 1 at zero recall must not win the F1 argmax.
    assert float(result["f1"][-1]) == 0.0
    assert float(result["confidence_at_max_f1"]) == 0.0


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


def test_update_after_reset_inside_inference_mode() -> None:
    """States reset inside C{torch.inference_mode} must still be
    updatable outside of it.

    The evaluation loop resets the metric under C{torch.inference_mode},
    which turns the default states into inference tensors. The
    quantization evaluation trainer then runs with
    C{inference_mode=False}, and the in-place accumulation used to fail
    with C{RuntimeError: Inplace update to inference tensor outside
    InferenceMode is not allowed}.
    """
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    predictions = make_pre_nms([(10, 10, 30, 30, 0.9, 0)])
    targets = torch.tensor(
        [[0, 0, 0.1, 0.1, 0.2, 0.2]],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        metric.reset()

    assert metric.true_positives.is_inference()

    metric.update(predictions, targets)

    torch.testing.assert_close(metric.true_positives, torch.tensor([1, 1, 0]))
    assert int(metric.target_count) == 1


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


def _record_nms(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    import luxonis_train.attached_modules.metrics.precision_recall_curve as module

    calls: list[dict[str, Any]] = []
    original = module.non_max_suppression

    def recorded_nms(*args, **kwargs) -> list[Tensor]:
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "non_max_suppression", recorded_nms)
    return calls


def test_precision_recall_curve_runs_nms_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_nms(monkeypatch)

    metric = make_metric()
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )

    assert len(calls) == 1


def test_nms_confidence_floor_is_positive_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NMS must not be re-run with a zero confidence threshold.

    The default grid starts at C{0.0}, and passing that straight to
    C{non_max_suppression} keeps every decoded anchor as an NMS
    candidate, which is orders of magnitude slower than the head's own
    post-processing. The floor is a separate parameter that defaults to
    C{1e-3}.
    """
    calls = _record_nms(monkeypatch)

    metric = PrecisionRecallCurve(node=make_node(), num_thresholds=11)
    assert metric.lowest_threshold == 0.0
    assert metric.nms_conf_threshold == pytest.approx(1e-3)

    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )

    assert calls[0]["conf_thres"] == pytest.approx(1e-3)
    assert calls[0]["conf_thres"] < 1e-3


def test_nms_confidence_floor_follows_lowest_threshold() -> None:
    metric = make_metric(thresholds=[0.4, 0.6, 0.8])

    assert metric.nms_conf_threshold == pytest.approx(0.4)


def test_prediction_at_lowest_threshold_is_counted() -> None:
    """A prediction scoring exactly the lowest threshold belongs to the
    first curve point.

    The curve counts are computed with C{>=} while
    C{non_max_suppression} filters with a strict C{>}, so a prediction
    scoring exactly the confidence floor used to be dropped before it
    could be counted.
    """
    metric = make_metric(thresholds=[0.5, 0.75, 0.9], nms_conf_threshold=0.5)
    assert metric.nms_conf_threshold == 0.5

    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.5, 0)]),
        torch.tensor([[0, 0, 0.1, 0.1, 0.2, 0.2]], dtype=torch.float32),
    )

    torch.testing.assert_close(metric.true_positives, torch.tensor([1, 0, 0]))
    assert float(metric.compute()["recall"][0]) == 1.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_exclusive_threshold_preserves_floor_in_low_precision(
    dtype: torch.dtype,
) -> None:
    """The NMS threshold predecessor must use the prediction dtype.

    A float32 predecessor can round back to 0.5 in low precision.
    """
    score = torch.tensor(0.5, dtype=dtype)

    assert score > _exclusive_threshold(0.5, dtype)


def test_greedy_matching_is_class_aware_and_uses_each_target_once() -> None:
    """Matching is greedy by score, per class, and every target may only
    be claimed once.
    """
    metric = make_metric(thresholds=[0.0, 0.65, 1.0])
    predictions = make_pre_nms(
        [
            (0, 0, 20, 10, 0.9, 0),
            (0, 10, 20, 20, 0.7, 0),
            (50, 50, 70, 70, 0.6, 1),
            (0, 0, 20, 20, 0.5, 1),
        ]
    )
    targets = torch.tensor(
        [
            [0, 0, 0.0, 0.0, 0.2, 0.2],
            [0, 1, 0.5, 0.5, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )

    metric.update(predictions, targets)

    # The 0.7 prediction overlaps the same target as the 0.9 one, and the
    # 0.5 prediction has the right class but no overlapping target.
    torch.testing.assert_close(metric.true_positives, torch.tensor([2, 1, 0]))
    torch.testing.assert_close(metric.false_positives, torch.tensor([2, 1, 0]))
    assert int(metric.target_count) == 2


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
        {"nms_conf_threshold": 1.5},
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_thresholds": 11},
        {"min_confidence": 0.5},
        {"max_confidence": 0.9},
        {"num_thresholds": 11, "min_confidence": 0.5},
    ],
)
def test_explicit_grid_rejects_generated_grid_parameters(
    kwargs: dict[str, Any],
) -> None:
    """An explicit grid combined with the generated-grid parameters is a
    configuration error.

    The explicit grid used to silently win, so a config carrying both
    produced a curve that ignored C{num_thresholds},
    C{min_confidence} and C{max_confidence} without any warning - and
    skipped their validation altogether.
    """
    with pytest.raises(ValueError, match="must not be combined"):
        PrecisionRecallCurve(
            node=make_node(),
            confidence_thresholds=[0.0, 0.5, 1.0],
            **kwargs,
        )


def test_confidence_axes_span_the_configured_grid() -> None:
    """The confidence axes must follow the threshold grid.

    Hardcoding C{xlim=(0, 1)} squeezed the whole curve into a sliver of
    the panel for a grid that only covers the high-confidence region.
    """
    metric = make_metric(thresholds=[0.9, 0.95, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.95, 0)]),
        torch.tensor([[0, 0, 0.1, 0.1, 0.2, 0.2]], dtype=torch.float32),
    )

    figure = metric.build_curve_figure(metric.compute())
    precision_recall, precision_conf, recall_conf = figure.axes

    assert precision_recall.get_xlim() == (0, 1)
    assert precision_conf.get_xlim() == pytest.approx((0.9, 1.0))
    assert recall_conf.get_xlim() == pytest.approx((0.9, 1.0))
    for axis in figure.axes:
        assert axis.get_ylim() == (0, 1)


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
    sanity_checking: bool = False,
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
            sanity_checking=sanity_checking,
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


def make_updated_metric() -> PrecisionRecallCurve:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    metric.update(
        make_pre_nms([(10, 10, 30, 30, 0.9, 0)]),
        torch.tensor(
            [[0, 0, 0.1, 0.1, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    )
    return metric


def test_precision_recall_curve_separates_scalars_and_artifact() -> None:
    metric = make_updated_metric()

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
    metric = make_updated_metric()
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
        "val/metrics/head/PrecisionRecallCurve/curves"
    )
    assert harness.tracker.images[0]["step"] == 3
    assert int(metric.target_count) == 0
    torch.testing.assert_close(
        metric.true_positives,
        torch.zeros(3, dtype=torch.long),
    )


def test_logged_artifact_path_matches_registered_mlflow_key() -> None:
    """The image name must resolve to the artifact key that
    C{get_mlflow_logging_keys} advertises.

    C{LuxonisTracker.log_image} inserts the step as its own path segment
    for MLflow, so a name that carries the epoch itself resolved to
    C{.../{epoch}/<node>/<metric>/{epoch}/curves.png} while the
    registered key had no step segment at all - every consumer resolving
    the advertised key found nothing.
    """
    metric = make_updated_metric()
    harness = make_epoch_end_harness(metric, is_global_zero=True)

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )
    keys = LuxonisLightningModule.get_mlflow_logging_keys(
        cast(LuxonisLightningModule, harness)
    )

    logged = harness.tracker.images[0]
    assert (
        mlflow_image_key(cast(str, logged["name"]), cast(int, logged["step"]))
        == "val/metrics/head/PrecisionRecallCurve/3/curves.png"
    )

    # The harness validates every epoch, so epoch 3 is registered too.
    harness.cfg.trainer.epochs = 4
    keys = LuxonisLightningModule.get_mlflow_logging_keys(
        cast(LuxonisLightningModule, harness)
    )
    assert (
        mlflow_image_key(cast(str, logged["name"]), cast(int, logged["step"]))
        in keys["artifacts"]
    )


def test_mlflow_image_key_matches_tracker_path_construction() -> None:
    """C{mlflow_image_key} mirrors the MLflow branch of
    C{LuxonisTracker.log_image}.
    """
    name = metric_artifact_image_name(
        "val", "head", "PrecisionRecallCurve", "curves"
    )
    assert name == "val/metrics/head/PrecisionRecallCurve/curves"

    base_path, caption = name.rsplit("/", 1)
    assert mlflow_image_key(name, 7) == f"{base_path}/7/{caption}.png"


def test_mlflow_keys_classify_curve_as_image_artifact() -> None:
    metric = make_metric(thresholds=[0.0, 0.5, 1.0])
    harness = make_epoch_end_harness(metric, is_global_zero=True)

    keys = LuxonisLightningModule.get_mlflow_logging_keys(
        cast(LuxonisLightningModule, harness)
    )

    assert "val/metric/head/PrecisionRecallCurve" in keys["metrics"]
    assert "val/metric/head/confidence_at_max_f1" in keys["metrics"]
    assert (
        "val/metrics/head/PrecisionRecallCurve/0/curves.png"
        in keys["artifacts"]
    )
    assert (
        "test/metrics/head/PrecisionRecallCurve/2/curves.png"
        in keys["artifacts"]
    )
    assert not any(
        name.endswith(("/precision", "/recall", "/confidence", "/f1"))
        for name in keys["metrics"]
    )


def test_evaluation_epoch_end_logs_primary_without_submetrics() -> None:
    metric = make_updated_metric()
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
    metric = make_updated_metric()
    harness = make_epoch_end_harness(metric, is_global_zero=False)

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )

    assert harness.tracker.images == []
    assert int(metric.target_count) == 0


def test_evaluation_epoch_end_skips_artifact_during_sanity_check() -> None:
    """Sanity checking must not publish a curve.

    The sanity-check validation pass runs before any epoch has finished,
    so logging its figure would publish a plot built from a couple of
    batches - and pay the full rendering cost on every run. Every other
    conditional logging path in the module already guards
    C{trainer.sanity_checking}.
    """
    metric = make_updated_metric()
    harness = make_epoch_end_harness(
        metric, is_global_zero=True, sanity_checking=True
    )

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )

    assert harness.tracker.images == []
    assert int(metric.target_count) == 0


def test_artifacts_are_built_before_the_metric_is_reset() -> None:
    """C{get_artifacts} must see the state the values were computed
    from.

    The artifact hook is public and may render from the metric's own
    accumulators, but it used to be called after C{metric.reset()} - so
    it silently rendered an all-zero figure next to correct scalars.
    """
    metric = make_updated_metric()
    observed: list[int] = []

    def get_artifacts(values: dict[str, Tensor]) -> dict[str, Tensor]:
        del values
        observed.append(int(metric.target_count))
        return {}

    metric.get_artifacts = get_artifacts  # type: ignore[method-assign]
    harness = make_epoch_end_harness(metric, is_global_zero=True)

    LuxonisLightningModule._evaluation_epoch_end(
        cast(LuxonisLightningModule, harness), "val"
    )

    assert observed == [1]
    assert int(metric.target_count) == 0


@pytest.mark.parametrize(
    "artifacts",
    [
        pytest.param({"curves": np.zeros((3, 4, 4))}, id="numpy-array"),
        pytest.param({"curves": object()}, id="not-a-tensor"),
        pytest.param({"curves": torch.zeros((10, 10))}, id="wrong-dim"),
        pytest.param([torch.zeros((3, 4, 4))], id="not-a-dict"),
        pytest.param(None, id="none"),
    ],
)
def test_log_metric_artifacts_tolerates_unusable_artifacts(
    artifacts: Any,
) -> None:
    """A metric returning something other than C{[C, H, W]} tensors must
    not take down the validation epoch.

    The dict iteration and the shape check used to sit outside both
    C{try} blocks, so a C{get_artifacts} returning e.g. a numpy image
    raised C{AttributeError} out of C{_evaluation_epoch_end} - defeating
    the whole point of handling artifact failures.
    """
    metric = make_updated_metric()
    metric.get_artifacts = lambda values: artifacts  # type: ignore[method-assign]
    tracker = DummyTracker()

    log_metric_artifacts(
        cast(Any, tracker),
        metric,
        metric.compute(),
        mode="val",
        formatted_node_name="head",
        metric_name="PrecisionRecallCurve",
        current_epoch=3,
    )

    assert tracker.images == []


def test_log_metric_artifacts_survives_generation_failure() -> None:
    metric = make_updated_metric()

    def failing_artifacts(values: dict[str, Tensor]) -> dict[str, Tensor]:
        del values
        raise RuntimeError("artifact rendering failed")

    metric.get_artifacts = failing_artifacts  # type: ignore[method-assign]
    tracker = DummyTracker()

    log_metric_artifacts(
        cast(Any, tracker),
        metric,
        metric.compute(),
        mode="val",
        formatted_node_name="head",
        metric_name="PrecisionRecallCurve",
        current_epoch=3,
    )

    assert tracker.images == []


def test_log_metric_artifacts_survives_tracker_failure() -> None:
    metric = make_updated_metric()

    class FailingTracker(DummyTracker):
        def log_image(self, name: str, img: object, step: int) -> None:
            raise RuntimeError("tracker image logging failed")

    tracker = FailingTracker()

    log_metric_artifacts(
        cast(Any, tracker),
        metric,
        metric.compute(),
        mode="val",
        formatted_node_name="head",
        metric_name="PrecisionRecallCurve",
        current_epoch=3,
    )

    assert tracker.images == []


def test_evaluation_epoch_end_survives_broken_artifacts() -> None:
    metric = make_updated_metric()
    metric.get_artifacts = lambda values: {"curves": np.zeros((3, 4, 4))}  # type: ignore[method-assign]
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


_REAL_BATCH_SIZE = 2
_REAL_IMAGE_HEIGHT = 64
_REAL_IMAGE_WIDTH = 64
_REAL_N_CLASSES = 2
_REAL_N_KEYPOINTS = 3
_REAL_FEATURE_SHAPES = (
    Size([_REAL_BATCH_SIZE, 16, 8, 8]),
    Size([_REAL_BATCH_SIZE, 32, 4, 4]),
    Size([_REAL_BATCH_SIZE, 64, 2, 2]),
)
_REAL_N_ANCHORS = sum(shape[-2] * shape[-1] for shape in _REAL_FEATURE_SHAPES)


class _FakeShapesMixin:
    @property
    def input_shapes(self) -> list[Packet[Size]]:
        return [{"features": list(_REAL_FEATURE_SHAPES)}]

    @property
    def in_sizes(self) -> list[Size]:
        return list(_REAL_FEATURE_SHAPES)


class _RealEfficientBBoxHead(
    _FakeShapesMixin, EfficientBBoxHead, register=False
):
    task = Tasks.BOUNDINGBOX
    original_in_shape = cast(
        Any, Size([3, _REAL_IMAGE_HEIGHT, _REAL_IMAGE_WIDTH])
    )
    n_classes = cast(Any, _REAL_N_CLASSES)


class _RealPrecisionBBoxHead(
    _FakeShapesMixin, PrecisionBBoxHead, register=False
):
    task = Tasks.BOUNDINGBOX
    original_in_shape = cast(
        Any, Size([3, _REAL_IMAGE_HEIGHT, _REAL_IMAGE_WIDTH])
    )
    n_classes = cast(Any, _REAL_N_CLASSES)


class _RealKeypointBBoxHead(
    _FakeShapesMixin, EfficientKeypointBBoxHead, register=False
):
    task = Tasks.INSTANCE_KEYPOINTS
    original_in_shape = cast(
        Any, Size([3, _REAL_IMAGE_HEIGHT, _REAL_IMAGE_WIDTH])
    )
    n_classes = cast(Any, _REAL_N_CLASSES)
    n_keypoints = cast(Any, _REAL_N_KEYPOINTS)


class _RealSegmentBBoxHead(
    _FakeShapesMixin, PrecisionSegmentBBoxHead, register=False
):
    task = Tasks.INSTANCE_SEGMENTATION
    original_in_shape = cast(
        Any, Size([3, _REAL_IMAGE_HEIGHT, _REAL_IMAGE_WIDTH])
    )
    n_classes = cast(Any, _REAL_N_CLASSES)


_REAL_HEADS = (
    _RealEfficientBBoxHead,
    _RealPrecisionBBoxHead,
    _RealKeypointBBoxHead,
    _RealSegmentBBoxHead,
)


def _make_real_features(seed: int) -> list[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(shape, generator=generator)
        for shape in _REAL_FEATURE_SHAPES
    ]


def _make_real_head(head_cls: type[BaseDetectionHead]) -> BaseDetectionHead:
    head = head_cls(
        n_heads=3,
        conf_thres=0.0,
        iou_thres=0.45,
        max_det=50,
        dataset_metadata=DatasetMetadata(
            classes={"": {"class0": 0, "class1": 1}},
            n_keypoints={"": _REAL_N_KEYPOINTS},
        ),
        task_name="",
    )
    head.eval()
    return head


@pytest.mark.parametrize("head_cls", _REAL_HEADS)
def test_pre_nms_candidates_are_opt_in(
    head_cls: type[BaseDetectionHead],
) -> None:
    """Heads only keep the pre-NMS candidates when asked to.

    The C{[B, n_anchors, 5 + n_classes]} tensor is large enough to shift
    peak validation memory, so it used to be paid for by every detection
    model whether or not a module consumed it.
    """
    torch.manual_seed(1300)
    head = _make_real_head(head_cls)

    assert head.keep_detections_pre_nms is False
    with torch.inference_mode():
        assert "detections_pre_nms" not in head(_make_real_features(1300))

    PrecisionRecallCurve(node=head, confidence_thresholds=[0.0, 0.5, 1.0])

    assert head.keep_detections_pre_nms is True
    with torch.inference_mode():
        packet = head(_make_real_features(1300))
    assert "detections_pre_nms" in packet
    detections_pre_nms = packet["detections_pre_nms"]
    assert isinstance(detections_pre_nms, Tensor)
    assert detections_pre_nms.shape[:2] == (_REAL_BATCH_SIZE, _REAL_N_ANCHORS)
    assert detections_pre_nms.shape[-1] >= 5 + _REAL_N_CLASSES


@pytest.mark.parametrize(
    ("head_cls", "seed"),
    [
        (_RealEfficientBBoxHead, 1301),
        (_RealPrecisionBBoxHead, 1302),
        (_RealKeypointBBoxHead, 1303),
        (_RealSegmentBBoxHead, 1304),
    ],
)
def test_real_bbox_head_to_precision_recall_curve_e2e(
    head_cls: type[BaseDetectionHead],
    seed: int,
) -> None:
    """Every detection head can feed the metric.

    The metric only supported C{BOUNDINGBOX} and only two of the four
    detection heads emitted the pre-NMS candidates, so attaching it to a
    keypoint or instance-segmentation head failed at model build time or
    at the first batch.
    """
    torch.manual_seed(seed)

    head = _make_real_head(head_cls)
    metric = PrecisionRecallCurve(
        node=head,
        confidence_thresholds=[0.0, 0.5, 1.0],
        matching_iou_threshold=0.5,
        nms_conf_threshold=0.0,
    )

    with torch.inference_mode():
        packet = head(_make_real_features(seed))

    assert {"boundingbox", "detections_pre_nms"} <= set(packet)

    detections_pre_nms = packet["detections_pre_nms"]
    boundingbox = packet["boundingbox"]

    assert isinstance(detections_pre_nms, Tensor)
    assert isinstance(boundingbox, list)
    assert detections_pre_nms.shape[:2] == (_REAL_BATCH_SIZE, _REAL_N_ANCHORS)
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
        torch.testing.assert_close(actual, expected[:, :6])

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
    assert score > 0
    torch.testing.assert_close(primary, computed["max_f1"])
    assert set(submetrics) == {"confidence_at_max_f1"}
    assert all(value.numel() == 1 for value in submetrics.values())

    artifact = artifacts["curves"]
    assert artifact.ndim == 3
    assert artifact.shape[0] == 3
    assert artifact.dtype == torch.uint8
