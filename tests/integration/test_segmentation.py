from typing import cast

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel
from luxonis_train.nodes.backbones import __all__ as BACKBONES

# TODO: Remove the following line after the incompatibility issue is resolved
BACKBONES = [backbone for backbone in BACKBONES if backbone != "PPLCNetV3"]


def get_opts(backbone: str) -> Params:
    opts = {
        "model": {
            "nodes": [
                {
                    "name": backbone,
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-color-segmentation",
                    "task_name": "color",
                    "inputs": [backbone],
                },
                {
                    "name": "BiSeNetHead",
                    "alias": "bi-color-segmentation",
                    "task_name": "color",
                    "inputs": [backbone],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation",
                    "task_name": "vehicles",
                    "inputs": [backbone],
                },
                {
                    "name": "BiSeNetHead",
                    "alias": "bi-vehicle-segmentation",
                    "task_name": "vehicles",
                    "inputs": [backbone],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation-2",
                    "task_name": "vehicles",
                    "inputs": [backbone],
                },
                {
                    "name": "SegmentationHead",
                    "alias": "seg-vehicle-segmentation-3",
                    "task_name": "vehicles",
                    "inputs": [backbone],
                },
            ],
            "losses": [
                {
                    "name": "CrossEntropyLoss",
                    "attached_to": "seg-color-segmentation",
                },
                {
                    "name": "CrossEntropyLoss",
                    "attached_to": "bi-color-segmentation",
                },
                {
                    "name": "BCEWithLogitsLoss",
                    "attached_to": "seg-vehicle-segmentation",
                },
                {
                    "name": "SigmoidFocalLoss",
                    "attached_to": "bi-vehicle-segmentation",
                    "params": {"alpha": 0.5, "gamma": 1.0},
                },
                {
                    "name": "SoftmaxFocalLoss",
                    "attached_to": "seg-vehicle-segmentation-2",
                    "params": {"alpha": 0.5, "gamma": 1.0},
                },
                {
                    "name": "SmoothBCEWithLogitsLoss",
                    "attached_to": "seg-vehicle-segmentation-3",
                    "params": {"label_smoothing": 0.1},
                },
            ],
            "metrics": [],
            "visualizers": [],
        }
    }
    aliases = [head["alias"] for head in opts["model"]["nodes"][1:]]
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


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbones(
    backbone: str, config: Params, parking_lot_dataset: LuxonisDataset
):
    opts = get_opts(backbone)
    opts["loader.params.dataset_name"] = parking_lot_dataset.identifier
    train_and_test(config, opts)
