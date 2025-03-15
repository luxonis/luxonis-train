from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Tasks

from .detection_confusion_matrix import DetectionConfusionMatrix
from .recognition_confusion_matrix import RecognitionConfusionMatrix
from .utils import preprocess_instance_masks


class InstanceSegmentationConfusionMatrix(
    DetectionConfusionMatrix, RecognitionConfusionMatrix
):
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

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
            *preprocess_instance_masks(
                boundingbox,
                instance_segmentation,
                target_boundingbox,
                target_instance_segmentation,
                self.n_classes,
                *self.original_in_shape[1:],
                device=self.device,
            ),
        )

    @override
    def compute(self) -> dict[str, Tensor]:
        return {
            "detection": DetectionConfusionMatrix.compute(self),
            "segmentation": RecognitionConfusionMatrix.compute(self),
        }
