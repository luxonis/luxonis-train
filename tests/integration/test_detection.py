from typing import Any

import pytest
from luxonis_ml.data import LuxonisDataset

from luxonis_train.core import LuxonisModel
from luxonis_train.nodes.backbones import __all__ as BACKBONES


def get_opts(backbone: str) -> dict[str, Any]:
    return {
        "model": {
            "nodes": [
                {
                    "name": backbone,
                },
                {
                    "name": "EfficientBBoxHead",
                    "inputs": [backbone],
                },
                {
                    "name": "EfficientKeypointBBoxHead",
                    "task": {
                        "keypoints": "car-keypoints",
                        "boundingbox": "car-boundingbox",
                    },
                    "inputs": [backbone],
                },
                {
                    "name": "ImplicitKeypointBBoxHead",
                    "task": {
                        "keypoints": "car-keypoints",
                        "boundingbox": "car-boundingbox",
                    },
                    "inputs": [backbone],
                },
            ],
            "losses": [
                {
                    "name": "AdaptiveDetectionLoss",
                    "attached_to": "EfficientBBoxHead",
                },
                {
                    "name": "EfficientKeypointBBoxLoss",
                    "attached_to": "EfficientKeypointBBoxHead",
                    "params": {"area_factor": 0.5},
                },
                {
                    "name": "ImplicitKeypointBBoxLoss",
                    "attached_to": "ImplicitKeypointBBoxHead",
                },
            ],
            "metrics": [
                {
                    "name": "MeanAveragePrecision",
                    "attached_to": "EfficientBBoxHead",
                },
                {
                    "name": "MeanAveragePrecisionKeypoints",
                    "alias": "EfficientKeypointBBoxHead-MaP",
                    "attached_to": "EfficientKeypointBBoxHead",
                },
                {
                    "name": "MeanAveragePrecisionKeypoints",
                    "alias": "ImplicitKeypointBBoxHead-MaP",
                    "attached_to": "ImplicitKeypointBBoxHead",
                },
            ],
        }
    }


def train_and_test(
    config: dict[str, Any],
    opts: dict[str, Any],
    train_overfit: bool = False,
):
    model = LuxonisModel(config, opts)
    model.train()
    results = model.test(view="val")
    if train_overfit:
        for name, value in results.items():
            if "/map_50" in name or "/kpt_map_medium" in name:
                assert value > 0.8, f"{name} = {value} (expected > 0.8)"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbones(
    backbone: str,
    config: dict[str, Any],
    parking_lot_dataset: LuxonisDataset,
):
    opts = get_opts(backbone)
    opts["loader.params.dataset_name"] = parking_lot_dataset.identifier
    train_and_test(config, opts)
