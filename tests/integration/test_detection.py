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


def get_opts_backbone(backbone: str) -> Params:
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
            ],
            "metrics": [
                {
                    "name": "MeanAveragePrecision",
                    "attached_to": "EfficientBBoxHead",
                },
                {
                    "name": "MeanAveragePrecision",
                    "alias": "EfficientKeypointBBoxHead-MaP",
                    "attached_to": "EfficientKeypointBBoxHead",
                },
            ],
        }
    }


def train_and_test(config: Params, opts: Params, train_overfit: bool = False):
    model = LuxonisModel(config, opts)
    model.train()
    if train_overfit:  # pragma: no cover
        results = model.test(view="val")
        for name, value in results.items():
            if "/map_50" in name or "/kpt_map_medium" in name:
                assert value > 0.8, f"{name} = {value} (expected > 0.8)"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbones(
    backbone: str, config: Params, coco_dataset: LuxonisDataset
):
    opts = get_opts_backbone(backbone)
    opts["loader.params.dataset_name"] = coco_dataset.identifier
    opts["trainer.epochs"] = 1
    train_and_test(config, opts)
