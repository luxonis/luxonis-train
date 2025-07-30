import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel
from luxonis_train.nodes.backbones import __all__ as BACKBONES

BACKBONES = [
    backbone
    for backbone in BACKBONES
    if backbone not in {"PPLCNetV3", "GhostFaceNetV2"}
]


def get_config(backbone: str) -> Params:
    return {
        "model": {
            "nodes": [
                {"name": backbone},
                {
                    "name": "EfficientBBoxHead",
                    "task_name": "vehicles",
                    "losses": [{"name": "AdaptiveDetectionLoss"}],
                    "metrics": [
                        {"name": "MeanAveragePrecision"},
                        {"name": "ConfusionMatrix"},
                    ],
                    "visualizers": [{"name": "BBoxVisualizer"}],
                },
                {
                    "name": "EfficientKeypointBBoxHead",
                    "task_name": "motorbikes",
                    "losses": [{"name": "EfficientKeypointBBoxLoss"}],
                    "metrics": [
                        {"name": "MeanAveragePrecision"},
                        {"name": "ConfusionMatrix"},
                    ],
                    "visualizers": [{"name": "KeypointVisualizer"}],
                },
                {
                    "name": "PrecisionSegmentBBoxHead",
                    "task_name": "vehicles",
                    "losses": [{"name": "PrecisionDFLSegmentationLoss"}],
                    "metrics": [
                        {"name": "MeanAveragePrecision"},
                        {"name": "ConfusionMatrix"},
                    ],
                    "visualizers": [
                        {"name": "InstanceSegmentationVisualizer"}
                    ],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-color-segmentation",
                    "task_name": "color",
                    "losses": [{"name": "CrossEntropyLoss"}],
                    "metrics": [{"name": "JaccardIndex"}, {"name": "F1Score"}],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "BiSeNetHead",
                    "alias": "bi-color-segmentation",
                    "task_name": "color",
                    "losses": [{"name": "CrossEntropyLoss"}],
                    "metrics": [{"name": "JaccardIndex"}, {"name": "F1Score"}],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation",
                    "task_name": "vehicles",
                    "losses": [{"name": "BCEWithLogitsLoss"}],
                    "metrics": [{"name": "JaccardIndex"}, {"name": "F1Score"}],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "BiSeNetHead",
                    "alias": "bi-vehicle-segmentation",
                    "task_name": "vehicles",
                    "losses": [{"name": "SigmoidFocalLoss"}],
                    "metrics": [{"name": "JaccardIndex"}, {"name": "F1Score"}],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation-2",
                    "task_name": "vehicles",
                    "losses": [{"name": "SoftmaxFocalLoss"}],
                    "metrics": [{"name": "JaccardIndex"}, {"name": "F1Score"}],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation-3",
                    "task_name": "vehicles",
                    "losses": [
                        {
                            "name": "SmoothBCEWithLogitsLoss",
                            "params": {"label_smoothing": 0.1},
                        }
                    ],
                    "metrics": [{"name": "JaccardIndex"}, {"name": "F1Score"}],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
            ],
        }
    }


def train_and_test(
    config: Params,
    opts: Params,
    train_overfit: bool = False,
):
    model = LuxonisModel(config, opts)
    model.train()
    if train_overfit:  # pragma: no cover
        results = model.test(view="val")
        for name, value in results.items():
            if "metric" in name:
                assert value > 0.8, f"{name} = {value} (expected > 0.8)"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbones(
    backbone: str, opts: Params, parking_lot_dataset: LuxonisDataset
):
    config = get_config(backbone)
    opts |= {
        "loader.params.dataset_name": parking_lot_dataset.identifier,
        "trainer.batch_size": 2,
    }
    model = LuxonisModel(config, opts)
    model.train()
    model.test()
    model.export()
    model.archive()
