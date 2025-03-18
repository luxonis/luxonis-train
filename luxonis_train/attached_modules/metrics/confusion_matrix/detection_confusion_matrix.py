import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks
from luxonis_train.utils.general import instances_from_batch


class DetectionConfusionMatrix(BaseMetric):
    supported_tasks = [
        Tasks.BOUNDINGBOX,
        Tasks.INSTANCE_KEYPOINTS,
        Tasks.INSTANCE_SEGMENTATION,
    ]

    confusion_matrix: Tensor

    def __init__(self, iou_threshold: float = 0.45, **kwargs):
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold

        self.add_state(
            "confusion_matrix",
            default=torch.zeros(
                self.n_classes + 1, self.n_classes + 1, dtype=torch.int64
            ),
            dist_reduce_fx="sum",
        )

    @override
    def update(
        self, boundingbox: list[Tensor], target_boundingbox: Tensor
    ) -> None:
        target_boundingbox[:, 2:] = box_convert(
            target_boundingbox[:, 2:], "xywh", "xyxy"
        )
        target_boundingbox[:, [2, 4]] *= self.original_in_shape[2]
        target_boundingbox[:, [3, 5]] *= self.original_in_shape[1]

        self._update(boundingbox, target_boundingbox)

    @override
    def compute(self) -> Tensor:
        return self.confusion_matrix

    def _update(self, predictions: list[Tensor], targets: Tensor) -> None:
        for pred, target in zip(
            predictions,
            instances_from_batch(targets, batch_size=len(predictions)),
            strict=True,
        ):
            pred_classes = pred[:, 5].int()
            target_classes = target[:, 0].int()

            # True Negatives
            if target.numel() == pred.numel() == 0:
                self.confusion_matrix[self.n_classes, self.n_classes] += 1

            # False Positives
            elif target.numel() == 0:
                self.confusion_matrix[pred_classes, self.n_classes] += 1

            # False Negatives
            elif pred.numel() == 0:
                self.confusion_matrix[self.n_classes, target_classes] += 1

            else:
                iou = box_iou(target[:, 1:], pred[:, :4])
                if (iou > self.iou_threshold).any():
                    iou_max, pred_max_idx = torch.max(iou, dim=1)
                    iou_target_mask = iou_max > self.iou_threshold
                    target_match_idx = torch.arange(
                        len(target), device=self.device
                    )[iou_target_mask]
                    pred_match_idx = pred_max_idx[iou_target_mask]

                    self.confusion_matrix.index_put_(
                        (
                            pred_classes[pred_match_idx],
                            target_classes[target_match_idx],
                        ),
                        torch.tensor(1),
                        accumulate=True,
                    )

                    for target_idx in self._get_unmatched(
                        target_match_idx, len(target)
                    ):
                        self.confusion_matrix[
                            self.n_classes, target_classes[target_idx]
                        ] += 1

                    for pred_idx in self._get_unmatched(
                        pred_match_idx, len(pred)
                    ):
                        self.confusion_matrix[
                            pred_classes[pred_idx], self.n_classes
                        ] += 1
                else:
                    self.confusion_matrix[self.n_classes, target_classes] += 1
                    self.confusion_matrix[pred_classes, self.n_classes] += 1

    def _get_unmatched(self, index: Tensor, size: int) -> Tensor:
        return torch.arange(size, device=self.device)[
            torch.isin(
                torch.arange(size, device=self.device),
                index,
                invert=True,
            )
        ]
