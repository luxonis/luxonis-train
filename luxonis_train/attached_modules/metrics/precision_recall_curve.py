from typing import cast

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from luxonis_train.tasks import Tasks
from luxonis_train.utils import non_max_suppression

from .base_metric import BaseMetric


class PrecisionRecallCurve(BaseMetric):
    """Compute global detection curves over a fixed confidence grid."""

    supported_tasks = [Tasks.BOUNDINGBOX]
    thresholds: Tensor
    true_positives: Tensor
    false_positives: Tensor
    target_count: Tensor

    def __init__(
        self,
        *,
        confidence_thresholds: list[float] | None = None,
        num_thresholds: int = 101,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        matching_iou_threshold: float = 0.5,
        nms_iou_threshold: float | None = None,
        max_detections: int | None = None,
        **kwargs,
    ):
        """Initialize the detection curve metric.

        @type confidence_thresholds: list[float] | None
        @param confidence_thresholds: Explicit confidence grid. When
            omitted, a linearly spaced grid is created.
        @type num_thresholds: int
        @param num_thresholds: Number of points in the generated grid.
        @type min_confidence: float
        @param min_confidence: Minimum generated confidence threshold.
        @type max_confidence: float
        @param max_confidence: Maximum generated confidence threshold.
        @type matching_iou_threshold: float
        @param matching_iou_threshold: IoU required for a prediction to
            match a ground-truth box.
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
        self.min_confidence = float(thresholds[0])

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
        num_thresholds: int,
        min_confidence: float,
        max_confidence: float,
    ) -> Tensor:
        if confidence_thresholds is not None:
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
            value = cast(float, self.node.iou_thres)
        return self._validate_unit_interval(value, "nms_iou_threshold")

    def _resolve_max_detections(self, value: int | None) -> int:
        if value is None:
            value = cast(int, self.node.max_det)
        if value <= 0:
            raise ValueError("max_detections must be positive.")
        return int(value)

    def update(
        self,
        detections_pre_nms: Tensor,
        target_boundingbox: Tensor,
    ) -> None:
        """Update curve counts from one batch of decoded candidates."""
        target_boundingbox = target_boundingbox.to(detections_pre_nms.device)
        predictions = non_max_suppression(
            detections_pre_nms,
            n_classes=self.node.n_classes,
            conf_thres=self.min_confidence,
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

            scores = image_predictions[:, 4]
            prediction_boxes = image_predictions[:, :4]
            prediction_classes = image_predictions[:, 5].long()
            true_positive = torch.zeros(
                len(image_predictions),
                dtype=torch.bool,
                device=image_predictions.device,
            )
            matched_targets = torch.zeros(
                len(target_boxes),
                dtype=torch.bool,
                device=image_predictions.device,
            )

            for prediction_index in range(len(image_predictions)):
                candidate_indices = torch.where(
                    (target_classes == prediction_classes[prediction_index])
                    & ~matched_targets
                )[0]
                if candidate_indices.numel() == 0:
                    continue

                ious = box_iou(
                    prediction_boxes[prediction_index].unsqueeze(0),
                    target_boxes[candidate_indices],
                ).squeeze(0)
                best_iou, best_local_index = torch.max(ious, dim=0)

                if best_iou >= self.matching_iou_threshold:
                    target_index = candidate_indices[best_local_index]
                    matched_targets[target_index] = True
                    true_positive[prediction_index] = True

            all_scores.append(scores)
            all_true_positive.append(true_positive)

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
            torch.zeros_like(true_positives),
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
        from matplotlib.figure import Figure

        from luxonis_train.attached_modules.visualizers.utils import (
            figure_to_torch,
        )

        confidence = values["confidence"].detach().cpu().numpy()
        precision = values["precision"].detach().cpu().numpy()
        recall = values["recall"].detach().cpu().numpy()

        artifact_width = 1200
        artifact_height = 400
        figure = Figure(
            figsize=(artifact_width / 100, artifact_height / 100),
            layout="constrained",
        )
        axes = figure.subplots(1, 3)

        axes[0].plot(recall[::-1], precision[::-1])
        axes[0].set_title("Precision-Recall")
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")

        axes[1].plot(confidence, precision)
        axes[1].set_title("Precision-Confidence")
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Precision")

        axes[2].plot(confidence, recall)
        axes[2].set_title("Recall-Confidence")
        axes[2].set_xlabel("Confidence")
        axes[2].set_ylabel("Recall")

        for axis in axes:
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid()

        return {
            "curves": figure_to_torch(
                figure,
                width=artifact_width,
                height=artifact_height,
            )
        }
