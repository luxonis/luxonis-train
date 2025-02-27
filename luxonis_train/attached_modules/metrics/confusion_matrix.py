import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)
from torchvision.ops import box_convert, box_iou
from typing_extensions import override

from luxonis_train.tasks import InstanceBaseTask, Metadata, Task, Tasks

from .base_metric import BaseMetric


# TODO: Possibly split to two classes and treat `ConfusionMatrix` as
# a factory with `__new__` method.
class ConfusionMatrix(BaseMetric):
    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.BOUNDINGBOX,
        Tasks.SEGMENTATION,
        Tasks.INSTANCE_SEGMENTATION,
        Tasks.INSTANCE_KEYPOINTS,
    ]

    def __init__(
        self,
        **kwargs,
    ):
        """Compute the confusion matrix for classification,
        segmentation, and object detection tasks."""
        super().__init__(**kwargs)
        self.metric_cm: Metric | None = None
        self.detection_cm: Tensor | None = None

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

        if self.task in {Tasks.BOUNDINGBOX, Tasks.INSTANCE_KEYPOINTS}:
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
        if isinstance(task, InstanceBaseTask):
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

        if self.detection_cm is not None:
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

        # TODO: Could be unified?
        if self.task == Tasks.CLASSIFICATION:
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

        assert self.metric_cm is not None
        self.metric_cm.update(preds.view(-1), targets.view(-1))

    def compute(self) -> dict[str, Tensor]:
        """Compute confusion matrices for classification, segmentation,
        and detection tasks."""
        results = {}
        if self.metric_cm is not None:
            results[f"{self.task}_confusion_matrix"] = self.metric_cm.compute()
        if self.detection_cm is not None:
            results["detection_confusion_matrix"] = self.detection_cm

        return results

    def reset(self) -> None:
        if self.metric_cm is not None:
            self.metric_cm.reset()

        if self.detection_cm is not None:
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

            pred_boxes = pred[:, :4]
            pred_classes = pred[:, 5].int()

            gt_boxes = img_targets[:, 2:]
            gt_classes = img_targets[:, 1].int()

            iou = box_iou(gt_boxes, pred_boxes)

            if iou.any():
                _, pred_max_idx = torch.max(iou, dim=1)
                gt_match_idx = torch.arange(
                    len(gt_boxes), device=gt_boxes.device
                )

                for gt_idx, pred_idx in zip(gt_match_idx, pred_max_idx):
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
                    pred_max_idx,
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
