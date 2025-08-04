import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params, ParamValue
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel
from luxonis_train.nodes.backbones import __all__ as BACKBONES

BACKBONES = [
    backbone
    for backbone in BACKBONES
    if backbone not in {"PPLCNetV3", "GhostFaceNetV2"}
]


def get_config(backbone: str) -> Params:
    seg_multi_losses: ParamValue = [
        {"name": "CrossEntropyLoss"},
        {"name": "SigmoidFocalLoss"},
        {"name": "OHEMLoss"},
    ]
    seg_binary_losses: ParamValue = [
        {"name": "BCEWithLogitsLoss"},
        {
            "name": "SmoothBCEWithLogitsLoss",
            "params": {"label_smoothing": 0.1},
        },
        {"name": "SigmoidFocalLoss"},
        {"name": "OHEMLoss"},
    ]
    seg_metrics: ParamValue = [
        {"name": "JaccardIndex"},
        {"name": "F1Score"},
        {"name": "Accuracy"},
        {"name": "Precision"},
        {"name": "Recall"},
        {"name": "ConfusionMatrix"},
    ]
    return {
        "model": {
            "nodes": [
                {"name": backbone},
                {
                    "name": "EfficientBBoxHead",
                    "task_name": "vehicles",
                    "losses": [
                        {
                            "name": "AdaptiveDetectionLoss",
                            "params": {"per_class_weights": [0, 0.5, 0.5]},
                        }
                    ],
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
                    "name": "BiSeNetHead",
                    "alias": "BiSeNet-binary-cars",
                    "task_name": "cars",
                    "losses": seg_binary_losses,
                    "metrics": seg_metrics,
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "BiSeNetHead",
                    "alias": "BiSeNet-multi-color",
                    "task_name": "color",
                    "losses": seg_multi_losses,
                    "metrics": seg_metrics,
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-binary-motorbikes",
                    "task_name": "motorbikes",
                    "losses": seg_binary_losses,
                    "metrics": seg_metrics,
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-multi-vehicles",
                    "task_name": "vehicles",
                    "losses": seg_multi_losses,
                    "metrics": seg_metrics,
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
            ],
        }
    }


@pytest.mark.parametrize("backbone", BACKBONES)
def test_combinations(
    backbone: str,
    parking_lot_dataset: LuxonisDataset,
    opts: Params,
    subtests: SubTests,
):
    config = get_config(backbone)
    opts |= {"loader.params.dataset_name": parking_lot_dataset.identifier}
    model = LuxonisModel(config, opts)

    with subtests.test("train"):
        model.train()

    with subtests.test("test"):
        model.test()

    with subtests.test("export"):
        model.export()

    with subtests.test("archive"):
        model.archive()
