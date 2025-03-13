import torch
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Tasks

from .detection_confusion_matrix import DetectionConfusionMatrix
from .recognition_confusion_matrix import RecognitionConfusionMatrix


class InstanceConfusionMatrix(
    RecognitionConfusionMatrix, DetectionConfusionMatrix
):
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def update(
        self,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> None:
        DetectionConfusionMatrix.update(self, boundingbox, target_boundingbox)
        RecognitionConfusionMatrix.update(
            self,
            self._merge_predicted_masks(boundingbox, instance_segmentation),
            self._merge_target_masks(
                target_boundingbox,
                target_instance_segmentation,
                len(boundingbox),
            ),
        )

    @override
    def compute(self) -> dict[str, Tensor]:
        return DetectionConfusionMatrix.compute(
            self
        ) | RecognitionConfusionMatrix.compute(self)

    @override
    def reset(self) -> None:
        DetectionConfusionMatrix.reset(self)
        RecognitionConfusionMatrix.reset(self)

    def _merge_predicted_masks(
        self,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
    ) -> Tensor:
        mask = torch.zeros(
            len(boundingbox),
            self.n_classes,
            *self.original_in_shape[1:],
            dtype=torch.bool,
            device=instance_segmentation[0].device,
        )
        for i, (bboxes, segs) in enumerate(
            zip(boundingbox, instance_segmentation, strict=True)
        ):
            for j, seg in enumerate(segs):
                class_id = bboxes[j][5:].argmax()
                mask[i][class_id] |= seg.bool()

        return mask.to(instance_segmentation[0].dtype)

    def _merge_target_masks(
        self,
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
        batch_size: int,
    ) -> Tensor:
        mask = torch.zeros(
            batch_size,
            self.n_classes,
            *self.original_in_shape[1:],
            dtype=torch.bool,
            device=target_instance_segmentation.device,
        )
        for bboxes, segs in zip(
            target_boundingbox, target_instance_segmentation, strict=True
        ):
            batch_idx = bboxes[0].int()
            class_id = bboxes[1].int()
            mask[batch_idx][class_id] |= segs.bool()

        return mask.to(target_instance_segmentation.dtype)
