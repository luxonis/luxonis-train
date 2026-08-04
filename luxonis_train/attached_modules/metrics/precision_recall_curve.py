from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from luxonis_train.nodes.heads.base_detection_head import BaseDetectionHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import non_max_suppression

from .base_metric import BaseMetric

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT_NUM_THRESHOLDS = 101
DEFAULT_MIN_CONFIDENCE = 0.0
DEFAULT_MAX_CONFIDENCE = 1.0

ARTIFACT_WIDTH = 1200
ARTIFACT_HEIGHT = 400


class PrecisionRecallCurve(BaseMetric):
    """Compute global detection curves over a fixed confidence grid."""

    supported_tasks = [
        Tasks.BOUNDINGBOX,
        Tasks.INSTANCE_KEYPOINTS,
        Tasks.INSTANCE_SEGMENTATION,
    ]

    node: BaseDetectionHead
    thresholds: Tensor
    true_positives: Tensor
    false_positives: Tensor
    target_count: Tensor

    def __init__(
        self,
        *,
        confidence_thresholds: list[float] | None = None,
        num_thresholds: int | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        matching_iou_threshold: float = 0.5,
        nms_conf_threshold: float = 1e-3,
        nms_iou_threshold: float | None = None,
        max_detections: int | None = None,
        **kwargs,
    ):
        """Initialize the detection curve metric.

        @type confidence_thresholds: list[float] | None
        @param confidence_thresholds: Explicit confidence grid. Mutually
            exclusive with C{num_thresholds}, C{min_confidence} and
            C{max_confidence}. When omitted, a linearly spaced grid is
            created from those three parameters.
        @type num_thresholds: int | None
        @param num_thresholds: Number of points in the generated grid.
            Defaults to C{101}.
        @type min_confidence: float | None
        @param min_confidence: Minimum generated confidence threshold.
            Defaults to C{0.0}.
        @type max_confidence: float | None
        @param max_confidence: Maximum generated confidence threshold.
            Defaults to C{1.0}.
        @type matching_iou_threshold: float
        @param matching_iou_threshold: IoU required for a prediction to
            match a ground-truth box.
        @type nms_conf_threshold: float
        @param nms_conf_threshold: Confidence floor applied before NMS.
            Candidates scoring below it are discarded and therefore never
            counted, not even at the lowest confidence threshold. The
            metric spends most of its time in NMS, whose cost grows with
            the number of candidates above the floor, so a floor of C{0}
            keeps every decoded anchor scoring above zero and makes
            validation needlessly slow. The default of C{1e-3} matches
            common detection validation practice. The effective floor is
            the larger of this value and the lowest confidence
            threshold.
        @type nms_iou_threshold: float | None
        @param nms_iou_threshold: IoU used by NMS. Defaults to the
            attached detection head value.
        @type max_detections: int | None
        @param max_detections: Maximum detections retained by NMS.
            Defaults to the attached detection head value.
        """
        super().__init__(**kwargs)

        thresholds = self._build_thresholds(
            confidence_thresholds=confidence_thresholds,
            num_thresholds=num_thresholds,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
        )
        self.register_buffer("thresholds", thresholds, persistent=False)

        self.matching_iou_threshold = self._validate_unit_interval(
            matching_iou_threshold,
            "matching_iou_threshold",
        )
        self.nms_iou_threshold = self._resolve_nms_iou_threshold(
            nms_iou_threshold
        )
        self.max_detections = self._resolve_max_detections(max_detections)
        self.lowest_threshold = float(thresholds[0])
        self.nms_conf_threshold = max(
            self._validate_unit_interval(
                nms_conf_threshold, "nms_conf_threshold"
            ),
            self.lowest_threshold,
        )

        self.add_state(
            "true_positives",
            default=torch.zeros(len(thresholds), dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_positives",
            default=torch.zeros(len(thresholds), dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "target_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

        # The pre-NMS candidates are large; heads only keep them in
        # their output packet when a module such as this one asks for it.
        self.node.request_detections_pre_nms()

    @staticmethod
    def _validate_unit_interval(value: float, name: str) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
        return value

    @classmethod
    def _build_thresholds(
        cls,
        *,
        confidence_thresholds: list[float] | None,
        num_thresholds: int | None,
        min_confidence: float | None,
        max_confidence: float | None,
    ) -> Tensor:
        grid_parameters = {
            "num_thresholds": num_thresholds,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
        }
        if confidence_thresholds is not None:
            provided = sorted(
                name
                for name, value in grid_parameters.items()
                if value is not None
            )
            if provided:
                raise ValueError(
                    "confidence_thresholds must not be combined with "
                    f"{', '.join(provided)}; the explicit grid would "
                    "silently take precedence."
                )

            thresholds = torch.tensor(
                confidence_thresholds,
                dtype=torch.float32,
            )
            if thresholds.ndim != 1 or len(thresholds) < 2:
                raise ValueError(
                    "confidence_thresholds must contain at least two values."
                )
            if torch.any(thresholds < 0) or torch.any(thresholds > 1):
                raise ValueError(
                    "confidence_thresholds values must be between 0 and 1."
                )
            if not torch.all(thresholds[1:] > thresholds[:-1]):
                raise ValueError(
                    "confidence_thresholds must be strictly increasing."
                )
            return thresholds

        if num_thresholds is None:
            num_thresholds = DEFAULT_NUM_THRESHOLDS
        if min_confidence is None:
            min_confidence = DEFAULT_MIN_CONFIDENCE
        if max_confidence is None:
            max_confidence = DEFAULT_MAX_CONFIDENCE

        if num_thresholds < 2:
            raise ValueError("num_thresholds must be at least 2.")

        min_confidence = cls._validate_unit_interval(
            min_confidence,
            "min_confidence",
        )
        max_confidence = cls._validate_unit_interval(
            max_confidence,
            "max_confidence",
        )
        if min_confidence >= max_confidence:
            raise ValueError(
                "min_confidence must be lower than max_confidence."
            )

        return torch.linspace(
            min_confidence,
            max_confidence,
            num_thresholds,
            dtype=torch.float32,
        )

    def _resolve_nms_iou_threshold(self, value: float | None) -> float:
        if value is None:
            value = self.node.iou_thres
        return self._validate_unit_interval(value, "nms_iou_threshold")

    def _resolve_max_detections(self, value: int | None) -> int:
        if value is None:
            value = self.node.max_det
        if value <= 0:
            raise ValueError("max_detections must be positive.")
        return int(value)

    def update(
        self,
        detections_pre_nms: Tensor,
        target_boundingbox: Tensor,
    ) -> None:
        """Update curve counts from one batch of decoded candidates."""
        self._detach_inference_states()

        target_boundingbox = target_boundingbox.to(detections_pre_nms.device)
        predictions = non_max_suppression(
            detections_pre_nms,
            n_classes=self.node.n_classes,
            conf_thres=_exclusive_threshold(
                self.nms_conf_threshold,
                detections_pre_nms.dtype,
            ),
            iou_thres=self.nms_iou_threshold,
            bbox_format="xyxy",
            max_det=self.max_detections,
            predicts_objectness=False,
        )

        scores, true_positive = self._match_predictions(
            predictions,
            target_boundingbox,
        )
        active = scores.unsqueeze(0) >= self.thresholds.unsqueeze(1)

        self.true_positives += (active & true_positive.unsqueeze(0)).sum(dim=1)
        self.false_positives += (active & ~true_positive.unsqueeze(0)).sum(
            dim=1
        )
        self.target_count += target_boundingbox.shape[0]

    def _detach_inference_states(self) -> None:
        """Replace inference-mode states with regular tensors.

        C{reset} runs inside the evaluation loop, which Lightning
        executes under C{torch.inference_mode}, so the fresh states
        become inference tensors. In-place updates to those raise a
        C{RuntimeError} once the metric is updated outside inference
        mode again, as the quantization evaluation trainer does.
        """
        if self.true_positives.is_inference():
            self.true_positives = self.true_positives.clone()
        if self.false_positives.is_inference():
            self.false_positives = self.false_positives.clone()
        if self.target_count.is_inference():
            self.target_count = self.target_count.clone()

    def _match_predictions(
        self,
        predictions: list[Tensor],
        target_boundingbox: Tensor,
    ) -> tuple[Tensor, Tensor]:
        all_scores: list[Tensor] = []
        all_true_positive: list[Tensor] = []

        height, width = self.node.original_in_shape[-2:]

        for image_index, image_predictions in enumerate(predictions):
            if image_predictions.numel() == 0:
                continue

            image_targets = target_boundingbox[
                target_boundingbox[:, 0].long() == image_index
            ]
            target_boxes, target_classes = self._prepare_targets(
                image_targets,
                height=height,
                width=width,
            )

            order = torch.argsort(
                image_predictions[:, 4],
                descending=True,
                stable=True,
            )
            image_predictions = image_predictions[order]

            all_scores.append(image_predictions[:, 4])
            all_true_positive.append(
                self._match_image_predictions(
                    prediction_boxes=image_predictions[:, :4],
                    prediction_classes=image_predictions[:, 5].long(),
                    target_boxes=target_boxes,
                    target_classes=target_classes,
                )
            )

        if not all_scores:
            return (
                target_boundingbox.new_empty((0,)),
                torch.empty(
                    0,
                    dtype=torch.bool,
                    device=target_boundingbox.device,
                ),
            )

        scores = torch.cat(all_scores)
        true_positive = torch.cat(all_true_positive)
        order = torch.argsort(scores, descending=True, stable=True)
        return scores[order], true_positive[order]

    def _match_image_predictions(
        self,
        *,
        prediction_boxes: Tensor,
        prediction_classes: Tensor,
        target_boxes: Tensor,
        target_classes: Tensor,
    ) -> Tensor:
        """Greedily assign score-sorted predictions to unique targets.

        The IoU matrix is computed once per image and the remaining loop
        only visits predictions that have at least one same-class target
        above the IoU threshold, which is a small fraction of the
        C{max_detections} predictions in practice.
        """
        true_positive = torch.zeros(
            len(prediction_boxes),
            dtype=torch.bool,
            device=prediction_boxes.device,
        )
        if len(target_boxes) == 0 or len(prediction_boxes) == 0:
            return true_positive

        ious = box_iou(prediction_boxes, target_boxes)
        eligible = (
            prediction_classes.unsqueeze(1) == target_classes.unsqueeze(0)
        ) & (ious >= self.matching_iou_threshold)
        matched_targets = torch.zeros(
            len(target_boxes),
            dtype=torch.bool,
            device=prediction_boxes.device,
        )

        candidates = eligible.any(dim=1).nonzero(as_tuple=True)[0]
        for prediction_index in candidates.tolist():
            available = eligible[prediction_index] & ~matched_targets
            if not bool(available.any()):
                continue

            target_index = int(
                torch.argmax(
                    ious[prediction_index].masked_fill(~available, -1)
                )
            )
            matched_targets[target_index] = True
            true_positive[prediction_index] = True

        return true_positive

    @staticmethod
    def _prepare_targets(
        image_targets: Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[Tensor, Tensor]:
        if image_targets.numel() == 0:
            return (
                image_targets.new_empty((0, 4)),
                torch.empty(
                    0,
                    dtype=torch.long,
                    device=image_targets.device,
                ),
            )

        boxes = box_convert(
            image_targets[:, 2:6],
            in_fmt="xywh",
            out_fmt="xyxy",
        )
        boxes[:, 0::2] *= width
        boxes[:, 1::2] *= height
        return boxes, image_targets[:, 1].long()

    def compute(self) -> dict[str, Tensor]:
        """Compute precision, recall, and F1 for every confidence
        point.
        """
        true_positives = self.true_positives.float()
        false_positives = self.false_positives.float()
        false_negatives = (self.target_count - self.true_positives).float()

        precision = torch.where(
            true_positives + false_positives > 0,
            true_positives / (true_positives + false_positives),
            # No prediction reaches the threshold, so nothing is
            # predicted incorrectly. Reporting 0 instead would drag the
            # curve to the floor over the whole high-confidence range.
            torch.ones_like(true_positives),
        )
        recall = torch.where(
            self.target_count > 0,
            true_positives / (true_positives + false_negatives),
            torch.zeros_like(true_positives),
        )
        f1 = torch.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            torch.zeros_like(precision),
        )

        best_index = torch.argmax(f1)

        return {
            "confidence": self.thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "max_f1": f1[best_index],
            "confidence_at_max_f1": self.thresholds[best_index],
        }

    def get_loggable_values(
        self,
        values: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Use maximum F1 as the primary scalar metric."""
        return values["max_f1"], {
            "confidence_at_max_f1": values["confidence_at_max_f1"],
        }

    def get_artifact_names(self) -> tuple[str, ...]:
        """Return the stable name of the combined curve figure."""
        return ("curves",)

    def get_artifacts(
        self,
        values: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Render PR, precision-confidence, and recall-confidence
        curves.
        """
        from luxonis_train.attached_modules.visualizers.utils import (
            figure_to_torch,
        )

        return {
            "curves": figure_to_torch(
                self.build_curve_figure(values),
                width=ARTIFACT_WIDTH,
                height=ARTIFACT_HEIGHT,
            )
        }

    def build_curve_figure(self, values: dict[str, Tensor]) -> "Figure":
        """Build the three-panel curve figure."""
        from matplotlib.figure import Figure

        confidence = values["confidence"].detach().cpu().numpy()
        precision = values["precision"].detach().cpu().numpy()
        recall = values["recall"].detach().cpu().numpy()

        figure = Figure(
            figsize=(ARTIFACT_WIDTH / 100, ARTIFACT_HEIGHT / 100),
            layout="constrained",
        )
        axes = figure.subplots(1, 3)

        axes[0].plot(recall[::-1], precision[::-1])
        axes[0].set_title("Precision-Recall")
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].set_xlim(0, 1)

        axes[1].plot(confidence, precision)
        axes[1].set_title("Precision-Confidence")
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Precision")

        axes[2].plot(confidence, recall)
        axes[2].set_title("Recall-Confidence")
        axes[2].set_xlabel("Confidence")
        axes[2].set_ylabel("Recall")

        # The confidence axes span the configured grid, which is not
        # necessarily the whole unit interval.
        for axis in axes[1:]:
            axis.set_xlim(float(confidence[0]), float(confidence[-1]))

        for axis in axes:
            axis.set_ylim(0, 1)
            axis.grid()

        return figure


def _exclusive_threshold(value: float, dtype: torch.dtype) -> float:
    """Convert an inclusive confidence floor to an exclusive one.

    C{non_max_suppression} keeps candidates scoring strictly above
    C{conf_thres}, while the curve counts use C{>=}. Compute the
    predecessor in the prediction dtype before returning it as a float,
    so it does not round back to C{value}. A floor of C{0} cannot
    include zero-scoring predictions.
    """
    if value <= 0.0:
        return 0.0
    return float(
        torch.nextafter(
            torch.tensor(value, dtype=dtype),
            torch.tensor(float("-inf"), dtype=dtype),
        )
    )
