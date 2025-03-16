import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks
from luxonis_train.utils.general import instances_from_batch


class DetectionConfusionMatrix(BaseMetric):
    supported_tasks = [Tasks.BOUNDINGBOX, Tasks.INSTANCE_KEYPOINTS]

    confusion_matrix: Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            if target.numel() == 0 and pred.numel() == 0:
                self.confusion_matrix[self.n_classes, self.n_classes] += 1

            # False Positives
            elif target.numel() == 0:
                self.confusion_matrix[pred_classes, self.n_classes] += 1

            # False Negatives
            elif pred.numel() == 0:
                self.confusion_matrix[self.n_classes, target_classes] += 1

            else:
                matched = box_iou(target[:, 1:], pred[:, :4]).argmax(dim=1)

                # True Positives
                self.confusion_matrix.index_put_(
                    [pred_classes[matched], target_classes],
                    torch.ones_like(matched),
                    accumulate=True,
                )

                # False Positives
                self.confusion_matrix[
                    self._not_matched(pred_classes, pred, matched),
                    self.n_classes,
                ] += 1

                # False Negatives
                self.confusion_matrix[
                    self.n_classes,
                    self._not_matched(target_classes, target, matched),
                ] += 1

    def _not_matched(
        self, classes: Tensor, tensor: Tensor, indices: Tensor
    ) -> Tensor:
        return classes[
            torch.isin(
                torch.arange(len(tensor), device=self.device),
                indices,
                invert=True,
            )
        ]
