from typing import Any

from luxonis_train.core import LuxonisModel


def get_opts_backbone(backbone: str) -> dict[str, Any]:
    return {
        "model": {
            "nodes": [
                {
                    "name": backbone,
                    "params": {"variant": "n"}
                    if backbone == "RecSubNet"
                    else {},
                },
                {
                    "name": "EfficientBBoxHead",
                    "task_name": "vehicle_type",
                    "inputs": [backbone],
                },
                {
                    "name": "EfficientKeypointBBoxHead",
                    "task_name": "car",
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


def get_opts_variant(variant: str) -> dict[str, Any]:
    return {
        "model": {
            "nodes": [
                {
                    "name": "EfficientRep",
                    "alias": "backbone",
                    "params": {"variant": variant},
                },
                {
                    "name": "RepPANNeck",
                    "alias": "neck",
                    "inputs": ["backbone"],
                    "params": {"variant": variant},
                },
                {
                    "name": "EfficientBBoxHead",
                    "task_name": "motorbike",
                    "inputs": ["neck"],
                },
            ],
            "losses": [
                {
                    "name": "AdaptiveDetectionLoss",
                    "attached_to": "EfficientBBoxHead",
                },
            ],
            "metrics": [
                {
                    "name": "MeanAveragePrecision",
                    "attached_to": "EfficientBBoxHead",
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
    if train_overfit:  # pragma: no cover
        results = model.test(view="val")
        for name, value in results.items():
            if "/map_50" in name or "/kpt_map_medium" in name:
                assert value > 0.8, f"{name} = {value} (expected > 0.8)"


# @pytest.mark.parametrize("backbone", BACKBONES)
# def test_backbones(
#     backbone: str,
#     config: dict[str, Any],
#     parking_lot_dataset: LuxonisDataset,
# ):
#     opts = get_opts_backbone(backbone)
#     opts["loader.params.dataset_name"] = parking_lot_dataset.identifier
#     opts["trainer.epochs"] = 1
#     train_and_test(config, opts)
#
#
# @pytest.mark.parametrize("variant", ["n", "s", "m", "l"])
# def test_variants(
#     variant: str,
#     config: dict[str, Any],
#     parking_lot_dataset: LuxonisDataset,
# ):
#     opts = get_opts_variant(variant)
#     opts["loader.params.dataset_name"] = parking_lot_dataset.identifier
#     opts["trainer.epochs"] = 1
#     train_and_test(config, opts)
