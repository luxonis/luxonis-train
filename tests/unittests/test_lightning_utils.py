from torch import Tensor

from luxonis_train import BaseNode
from luxonis_train.attached_modules.metrics import MeanAveragePrecision, MIoU
from luxonis_train.lightning.utils import _translate_predefined_metric_params
from luxonis_train.tasks import Task, Tasks


class DummyNode(BaseNode):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def forward(self, _: Tensor) -> Tensor: ...


def test_translate_predefined_metric_params_detection_map():
    params = _translate_predefined_metric_params(
        DummyNode(Tasks.BOUNDINGBOX),
        "MeanAveragePrecision",
        MeanAveragePrecision,
        {"per_class_metrics": True},
    )

    assert params == {"class_metrics": True}


def test_translate_predefined_metric_params_keypoint_map():
    params = _translate_predefined_metric_params(
        DummyNode(Tasks.INSTANCE_KEYPOINTS),
        "MeanAveragePrecision",
        MeanAveragePrecision,
        {"per_class_metrics": True},
    )

    assert params == {}


def test_translate_predefined_metric_params_segmentation_iou():
    params = _translate_predefined_metric_params(
        DummyNode(Tasks.SEGMENTATION),
        "MIoU",
        MIoU,
        {"num_classes": 3, "per_class_metrics": True},
    )

    assert params == {"num_classes": 3, "per_class": True}
