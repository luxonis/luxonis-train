import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Metadata, Tasks


class DetectionConfusionMatrix(BaseMetric):
    supported_tasks = [Tasks.BOUNDINGBOX, Tasks.INSTANCE_KEYPOINTS]

    confusion_matrix: Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state(
            "confusion_matrix",
            # +1 for background
            default=torch.zeros(
                self.n_classes + 1, self.n_classes + 1, dtype=torch.int64
            ),
            dist_reduce_fx="sum",
        )

    @property
    @override
    def required_labels(self) -> set[str | Metadata]:
        return super().required_labels - {"keypoints"}

    @override
    def update(self, predictions: list[Tensor], targets: Tensor) -> None:
        """Prepare data for classification, segmentation, and detection
        tasks.

        @type inputs: Packet[Tensor]
        @param inputs: The inputs to the model.
        @type labels: Labels
        @param labels: The ground-truth labels.
        @return: A tuple of two dictionaries: one for predictions and
            one for targets.
        """

        targets[..., 2:6] = box_convert(targets[..., 2:6], "xywh", "xyxy")
        scale_factors = torch.tensor(
            [
                self.original_in_shape[2],
                self.original_in_shape[1],
                self.original_in_shape[2],
                self.original_in_shape[1],
            ],
            device=targets.device,
        )
        targets[..., 2:6] *= scale_factors

        self.confusion_matrix += self._compute_detection_confusion_matrix(
            predictions, targets
        )

    @override
    def compute(self) -> dict[str, Tensor]:
        return {"detection_confusion_matrix": self.confusion_matrix}

    def _compute_detection_confusion_matrix(
        self, predictions: list[Tensor], targets: Tensor
    ) -> Tensor:
        """Compute a confusion matrix for object detection tasks.

        @type predictions: list[Tensor]
        @param predictions: List of predictions for each image. Each
            tensor has shape [N, 6] where 6 is for [x1, y1, x2, y2,
            score, class]
        @type targets: Tensor
        @param targets: Ground truth boxes and classes. Shape [M, 6]
            where first column is image index.
        """
        conf_matrix = torch.zeros(
            self.n_classes + 1,
            self.n_classes + 1,
            dtype=torch.int64,
            device=predictions[0].device,
        )

        for i, pred in enumerate(predictions):
            target = targets[targets[:, 0] == i]

            if target.numel() == 0:
                for pred_class in pred[:, 5].int():
                    conf_matrix[pred_class, self.n_classes] += 1
                continue

            if pred.numel() == 0:
                for target_class in target[:, 1].int():
                    conf_matrix[self.n_classes, target_class] += 1
                continue

            pred_boxes = pred[:, :4]
            pred_classes = pred[:, 5].int()

            target_boxes = target[:, 2:]
            target_classes = target[:, 1].int()

            iou = box_iou(target_boxes, pred_boxes)

            if iou.any():
                _, pred_max_idx = torch.max(iou, dim=1)
                target_match_idx = torch.arange(
                    len(target_boxes), device=target_boxes.device
                )

                for target_idx, pred_idx in zip(
                    target_match_idx, pred_max_idx, strict=True
                ):
                    target_class = target_classes[target_idx]
                    pred_class = pred_classes[pred_idx]
                    conf_matrix[pred_class, target_class] += 1

                unmatched_gt_mask = ~torch.isin(
                    torch.arange(
                        len(target_boxes), device=target_boxes.device
                    ),
                    target_match_idx,
                )
                for target_idx in torch.arange(
                    len(target_boxes), device=target_boxes.device
                )[unmatched_gt_mask]:
                    target_class = target_classes[target_idx]
                    conf_matrix[self.n_classes, target_class] += 1

                unmatched_pred_mask = ~torch.isin(
                    torch.arange(len(pred_boxes), device=target_boxes.device),
                    pred_max_idx,
                )
                for pred_idx in torch.arange(
                    len(pred_boxes), device=target_boxes.device
                )[unmatched_pred_mask]:
                    pred_class = pred_classes[pred_idx]
                    conf_matrix[pred_class, self.n_classes] += 1
            else:
                for target_class in target_classes:
                    conf_matrix[self.n_classes, target_class] += 1
                for pred_class in pred_classes:
                    conf_matrix[pred_class, self.n_classes] += 1

        return conf_matrix
