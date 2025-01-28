from typing import Literal

import torch
from torch import Tensor
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)
from torchvision.ops import box_convert, box_iou
from typing_extensions import override

from luxonis_train.tasks import Metadata, Task, Tasks

from .base_metric import BaseMetric


class ConfusionMatrix(BaseMetric):
    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.KEYPOINTS,
        Tasks.INSTANCE_SEGMENTATION,
        Tasks.BOUNDINGBOX,
    ]

    def __init__(
        self,
        box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
        iou_threshold: float = 0.45,
        confidence_threshold: float = 0.25,
        **kwargs,
    ):
        """Compute the confusion matrix for classification,
        segmentation, and object detection tasks.

        @type box_format: Literal["xyxy", "xywh", "cxcywh"]
        @param box_format: The format of the bounding boxes. Can be one
            of "xyxy", "xywh", or "cxcywh".
        @type iou_threshold: float
        @param iou_threshold: The IoU threshold for matching predictions
            to ground truth.
        @type confidence_threshold: float
        @param confidence_threshold: The confidence threshold for
            filtering predictions.
        """
        super().__init__(**kwargs)
        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}"
            )
        self.box_format = box_format
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        if self.task in {
            Tasks.CLASSIFICATION,
            Tasks.SEGMENTATION,
            Tasks.INSTANCE_SEGMENTATION,
        }:
            if self.n_classes == 1:
                self.metric_cm = BinaryConfusionMatrix()
            else:
                self.metric_cm = MulticlassConfusionMatrix(
                    num_classes=self.n_classes
                )

        self.detection_cm: Tensor

        if self.task in {Tasks.BOUNDINGBOX, Tasks.KEYPOINTS}:
            self.add_state(
                "detection_cm",
                default=torch.zeros(
                    self.n_classes + 1, self.n_classes + 1, dtype=torch.int64
                ),  # +1 for background
                dist_reduce_fx="sum",
            )

    @property
    @override
    def task(self) -> Task:
        task = super().task
        if task in {Tasks.KEYPOINTS, Tasks.INSTANCE_SEGMENTATION}:
            return Tasks.BOUNDINGBOX
        return task

    @property
    @override
    def required_labels(self) -> set[str | Metadata]:
        return super().required_labels - {"keypoints"}

    def update(
        self, predictions: Tensor | list[Tensor], target: Tensor
    ) -> None:
        """Prepare data for classification, segmentation, and detection
        tasks.

        @type inputs: Packet[Tensor]
        @param inputs: The inputs to the model.
        @type labels: Labels
        @param labels: The ground-truth labels.
        @return: A tuple of two dictionaries: one for predictions and
            one for targets.
        """

        if self.task in {Tasks.BOUNDINGBOX, Tasks.KEYPOINTS}:
            assert isinstance(predictions, list)
            target[..., 2:6] = box_convert(target[..., 2:6], "xywh", "xyxy")
            scale_factors = torch.tensor(
                [
                    self.original_in_shape[2],
                    self.original_in_shape[1],
                    self.original_in_shape[2],
                    self.original_in_shape[1],
                ],
                device=target.device,
            )
            target[..., 2:6] *= scale_factors

            self.detection_cm += self._compute_detection_confusion_matrix(
                predictions,
                target,
            )
            return

        assert isinstance(predictions, Tensor)

        if self.task is Tasks.CLASSIFICATION:
            preds = (
                predictions.argmax(dim=1)
                if predictions.shape[1] > 1
                else predictions.sigmoid().squeeze(1).round().int()
            )  # [B]
        else:
            assert isinstance(predictions, Tensor)
            preds = (
                predictions.argmax(dim=1)
                if predictions.shape[1] > 1
                else predictions.squeeze(1).sigmoid().round().int()
            )  # [B, H, W]

        targets = (
            target.argmax(dim=1)
            if target.shape[1] > 1
            else target.squeeze(1).round().int()
        )

        self.metric_cm.update(preds.view(-1), targets.view(-1))

    def compute(self) -> dict[str, Tensor]:
        """Compute confusion matrices for classification, segmentation,
        and detection tasks."""
        results = {}
        if hasattr(self, "metric_cm"):
            results[f"{self.task}_confusion_matrix"] = self.metric_cm.compute()
        if self.task in {Tasks.BOUNDINGBOX, Tasks.KEYPOINTS}:
            results["detection_confusion_matrix"] = self.detection_cm

        return results

    def reset(self) -> None:
        if hasattr(self, "metric_cm"):
            self.metric_cm.reset()

        if self.task in {Tasks.BOUNDINGBOX, Tasks.KEYPOINTS}:
            self.detection_cm.zero_()

    def _compute_detection_confusion_matrix(
        self, preds: list[Tensor], targets: Tensor
    ) -> Tensor:
        """Compute a confusion matrix for object detection tasks.

        @type preds: list[Tensor]
        @param preds: List of predictions for each image. Each tensor
            has shape [N, 6] where 6 is for [x1, y1, x2, y2, score,
            class]
        @type targets: Tensor
        @param targets: Ground truth boxes and classes. Shape [M, 6]
            where first column is image index.
        """
        cm = torch.zeros(
            self.n_classes + 1,
            self.n_classes + 1,
            dtype=torch.int64,
            device=preds[0].device,
        )

        for img_idx, pred in enumerate(preds):
            img_targets = targets[targets[:, 0] == img_idx]

            if img_targets.shape[0] == 0:
                for pred_class in pred[:, 5].int():
                    cm[pred_class, self.n_classes] += 1
                continue

            if pred.shape[0] == 0:
                for gt_class in img_targets[:, 1].int():
                    cm[self.n_classes, gt_class] += 1
                continue

            pred = pred[pred[:, 4] > self.confidence_threshold]
            pred_boxes = pred[:, :4]
            pred_classes = pred[:, 5].int()

            gt_boxes = img_targets[:, 2:]
            gt_classes = img_targets[:, 1].int()

            iou = box_iou(gt_boxes, pred_boxes)
            iou_thresholded = iou > self.iou_threshold

            if iou_thresholded.any():
                iou_max, pred_max_idx = torch.max(iou, dim=1)
                iou_gt_mask = iou_max > self.iou_threshold
                gt_match_idx = torch.arange(
                    len(gt_boxes), device=gt_boxes.device
                )[iou_gt_mask]
                pred_match_idx = pred_max_idx[iou_gt_mask]

                for gt_idx, pred_idx in zip(gt_match_idx, pred_match_idx):
                    gt_class = gt_classes[gt_idx]
                    pred_class = pred_classes[pred_idx]
                    cm[pred_class, gt_class] += 1

                unmatched_gt_mask = ~torch.isin(
                    torch.arange(len(gt_boxes), device=gt_boxes.device),
                    gt_match_idx,
                )
                for gt_idx in torch.arange(
                    len(gt_boxes), device=gt_boxes.device
                )[unmatched_gt_mask]:
                    gt_class = gt_classes[gt_idx]
                    cm[self.n_classes, gt_class] += 1

                unmatched_pred_mask = ~torch.isin(
                    torch.arange(len(pred_boxes), device=gt_boxes.device),
                    pred_match_idx,
                )
                for pred_idx in torch.arange(
                    len(pred_boxes), device=gt_boxes.device
                )[unmatched_pred_mask]:
                    pred_class = pred_classes[pred_idx]
                    cm[pred_class, self.n_classes] += 1
            else:
                for gt_class in gt_classes:
                    cm[self.n_classes, gt_class] += 1
                for pred_class in pred_classes:
                    cm[pred_class, self.n_classes] += 1

        return cm
