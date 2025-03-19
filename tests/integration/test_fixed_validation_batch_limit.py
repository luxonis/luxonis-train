from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def get_config() -> Params:
    return {
        "model": {
            "nodes": [
                {
                    "name": "EfficientRep",
                    "alias": "backbone",
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
            "name": "LuxonisLoaderTorch",
            "train_view": "train",
            "val_view": "train",
            "test_view": "train",
        },
        "trainer": {
            "n_validation_batches": 1,
        },
    }


def test_fixed_validation_batch_limit(parking_lot_dataset: LuxonisDataset):
    config = get_config()
    opts: Params = {
        "loader.params.dataset_name": parking_lot_dataset.identifier
    }
    model = LuxonisModel(config, opts)
    assert (
        len(model.pytorch_loaders["val"]) == 1
    ), "Validation loader should contain exactly 1 batch"
    assert (
        len(model.pytorch_loaders["test"]) == 1
    ), "Test loader should contain exactly 1 batch"
    opts["trainer.n_validation_batches"] = None
    model = LuxonisModel(config, opts)
    assert (
        len(model.pytorch_loaders["val"]) > 1
    ), "Validation loader should contain all validation samples"
    assert (
        len(model.pytorch_loaders["test"]) > 1
    ), "Test loader should contain all test samples"
