from typing import Literal

import torch
from torch import Tensor
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision.ops import box_convert, box_iou

from luxonis_train.enums import TaskType
from luxonis_train.utils import Labels, Packet

from .base_metric import BaseMetric


class ConfusionMatrix(BaseMetric[Tensor, Tensor]):
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

        self.is_classification = (
            self.node.tasks is not None
            and TaskType.CLASSIFICATION in self.node.tasks
        )
        self.is_detection = (
            self.node.tasks is not None
            and TaskType.BOUNDINGBOX in self.node.tasks
        )
        self.is_segmentation = (
            self.node.tasks is not None
            and TaskType.SEGMENTATION in self.node.tasks
        )

        if (
            sum(
                [
                    self.is_classification,
                    self.is_detection,
                    self.is_segmentation,
                ]
            )
            > 1
        ):
            raise ValueError(
                "Multiple tasks detected in self.node.tasks. Only one task is allowed."
            )

        self.metric_cm = None
        if self.is_classification or self.is_segmentation:
            self.metric_cm = MulticlassConfusionMatrix(
                num_classes=self.n_classes
            )

        if self.is_detection:
            self.add_state(
                "detection_cm",
                default=torch.zeros(
                    self.n_classes + 1, self.n_classes + 1, dtype=torch.int64
                ),  # +1 for background
                dist_reduce_fx="sum",
            )

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Prepare data for classification, segmentation, and detection
        tasks.

        @type inputs: Packet[Tensor]
        @param inputs: The inputs to the model.
        @type labels: Labels
        @param labels: The ground-truth labels.
        @return: A tuple of two dictionaries: one for predictions and
            one for targets.
        """
        predictions = {}
        targets = {}

        if self.is_detection:
            out_bbox = self.get_input_tensors(inputs, TaskType.BOUNDINGBOX)
            bbox = self.get_label(labels, TaskType.BOUNDINGBOX)
            bbox = bbox.to(out_bbox[0].device).clone()
            bbox[..., 2:6] = box_convert(bbox[..., 2:6], "xywh", "xyxy")
            scale_factors = torch.tensor(
                [
                    self.original_in_shape[2],
                    self.original_in_shape[1],
                    self.original_in_shape[2],
                    self.original_in_shape[1],
                ],
                device=bbox.device,
            )
            bbox[..., 2:6] *= scale_factors
            predictions["detection"] = out_bbox
            targets["detection"] = bbox

        if self.is_classification:
            prediction = self.get_input_tensors(
                inputs, TaskType.CLASSIFICATION
            )
            target = self.get_label(labels, TaskType.CLASSIFICATION).to(
                prediction[0].device
            )
            predictions["classification"] = prediction
            targets["classification"] = target

        if self.is_segmentation:
            prediction = self.get_input_tensors(inputs, TaskType.SEGMENTATION)
            target = self.get_label(labels, TaskType.SEGMENTATION).to(
                prediction[0].device
            )
            predictions["segmentation"] = prediction
            targets["segmentation"] = target

        return predictions, targets

    def update(
        self, predictions: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> None:
        """Update the confusion matrices for all tasks using prepared
        data.

        @type predictions: dict[str, Tensor]
        @param predictions: A dictionary containing predictions for all
            tasks.
        @type targets: dict[str, Tensor]
        @param targets: A dictionary containing targets for all tasks.
        """
        if "classification" in predictions and "classification" in targets:
            preds = predictions["classification"]
            target = targets["classification"]
            pred_classes = preds[0].argmax(dim=1)  # [B]
            target_classes = target.argmax(dim=1)  # [B]
            if self.metric_cm is not None:
                self.metric_cm.update(pred_classes, target_classes)

        if "segmentation" in predictions and "segmentation" in targets:
            preds = predictions["segmentation"]
            target = targets["segmentation"]
            pred_masks = preds[0].argmax(dim=1)  # [B, H, W]
            target_masks = target.argmax(dim=1)  # [B, H, W]
            if self.metric_cm is not None:
                self.metric_cm.update(
                    pred_masks.view(-1), target_masks.view(-1)
                )

        if "detection" in predictions and "detection" in targets:
            preds = predictions["detection"]  # type: ignore
            target = targets["detection"]
            self.detection_cm += self._compute_detection_confusion_matrix(  # type: ignore
                preds,  # type: ignore
                target,  # type: ignore
            )

    def compute(self) -> dict[str, Tensor]:
        """Compute confusion matrices for classification, segmentation,
        and detection tasks."""
        results = {}
        if self.metric_cm:
            task_type = (
                "classification" if self.is_classification else "segmentation"
            )
            results[f"{task_type}_confusion_matrix"] = self.metric_cm.compute()
            self.metric_cm.reset()
        if self.is_detection:
            results["detection_confusion_matrix"] = self.detection_cm

        return results

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
