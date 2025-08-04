from typing import cast

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


def get_opts(backbone: str, index: int) -> Params:
    opts = {
        "model": {
            "nodes": [
                {"name": backbone},
                {
                    "name": "SegmentationHead",
                    "alias": "seg-color-segmentation",
                    "task_name": "color",
                    "inputs": [backbone],
                    "losses": [{"name": "CrossEntropyLoss"}],
                }
                if index % 6 == 0
                else {
                    "name": "BiSeNetHead",
                    "alias": "bi-color-segmentation",
                    "task_name": "color",
                    "inputs": [backbone],
                    "losses": [{"name": "CrossEntropyLoss"}],
                }
                if index % 6 == 1
                else {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation",
                    "task_name": "vehicles",
                    "inputs": [backbone],
                    "losses": [{"name": "BCEWithLogitsLoss"}],
                }
                if index % 6 == 2
                else {
                    "name": "BiSeNetHead",
                    "alias": "bi-vehicle-segmentation",
                    "task_name": "vehicles",
                    "inputs": [backbone],
                    "losses": [
                        {
                            "name": "SigmoidFocalLoss",
                            "params": {"alpha": 0.5, "gamma": 1.0},
                        }
                    ],
                }
                if index % 6 == 3
                else {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation-2",
                    "task_name": "vehicles",
                    "inputs": [backbone],
                    "losses": [
                        {
                            "name": "SoftmaxFocalLoss",
                            "params": {"alpha": 0.5, "gamma": 1.0},
                        }
                    ],
                }
                if index % 6 == 4
                else {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation-3",
                    "task_name": "vehicles",
                    "inputs": [backbone],
                    "losses": [
                        {
                            "name": "SmoothBCEWithLogitsLoss",
                            "params": {"label_smoothing": 0.1},
                        }
                    ],
                },
            ],
        }
    }
    aliases = [head["alias"] for head in opts["model"]["nodes"][1:]]
    opts["model"]["metrics"] = []
    opts["model"]["visualizers"] = []
    for alias in aliases:
        opts["model"]["metrics"].extend(
            [
                {
                    "name": "JaccardIndex",
                    "alias": f"JaccardIndex_{alias}",
                    "attached_to": alias,
                },
                {
                    "name": "F1Score",
                    "alias": f"F1Score_{alias}",
                    "attached_to": alias,
                },
            ]
        )
        opts["model"]["visualizers"].append(
            {
                "name": "SegmentationVisualizer",
                "attached_to": alias,
            }
        )
    return cast(Params, opts)


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


@pytest.mark.parametrize(("index", "backbone"), enumerate(BACKBONES))
def test_backbones(
    index: int,
    backbone: str,
    config: Params,
    parking_lot_dataset: LuxonisDataset,
):
    opts = get_opts(backbone, index)
    opts["loader.params.dataset_name"] = parking_lot_dataset.identifier
    train_and_test(config, opts)
