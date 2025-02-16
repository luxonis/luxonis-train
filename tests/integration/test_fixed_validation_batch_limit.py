from typing import Any

from luxonis_ml.data import LuxonisDataset

from luxonis_train.core import LuxonisModel


def get_config() -> dict[str, Any]:
    return {
        "model": {
            "nodes": [
                {
                    "name": "EfficientRep",
                    "alias": "backbone",
                    "params": {"variant": "n"},
                },
                {
                    "name": "RepPANNeck",
                    "alias": "neck",
                    "inputs": ["backbone"],
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
                }
            ],
            "metrics": [
                {
                    "name": "MeanAveragePrecision",
                    "attached_to": "EfficientBBoxHead",
                }
            ],
        },
        "loader": {
            "name": "LuxonisTorchDataset",
            "train_splits": "train",
            "val_splits": "train",
            "test_splits": "train",
        },
        "trainer": {
            "n_validation_batches": 1,
        },
    }


def test_fixed_validation_batch_limit(parking_lot_dataset: LuxonisDataset):
    config = get_config()
    opts = {"loader.params.dataset_name": parking_lot_dataset.identifier}
    model = LuxonisModel(config, opts)
    assert (
        len(model.loaders["val"]) == 1
    ), "Validation loader should contain exactly 1 batch"
    assert (
        len(model.loaders["test"]) == 1
    ), "Test loader should contain exactly 1 batch"
    config["trainer"]["n_validation_batches"] = None
    model = LuxonisModel(config, opts)
    assert (
        len(model.loaders["val"]) > 1
    ), "Validation loader should contain all validation samples"
    assert (
        len(model.loaders["test"]) > 1
    ), "Test loader should contain all test samples"
