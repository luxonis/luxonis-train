from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def test_fixed_validation_batch_limit(
    parking_lot_dataset: LuxonisDataset, opts: Params
):
    cfg = "configs/detection_light_model.yaml"
    opts |= {
        "model.predefined_model.params.task_name": "vehicles",
        "loader.params.dataset_name": parking_lot_dataset.identifier,
        "loader.train_view": "train",
        "loader.val_view": "train",
        "loader.test_view": "train",
        "trainer.n_validation_batches": 1,
    }
    model = LuxonisModel(cfg, opts)
    assert len(model.pytorch_loaders["val"]) == 1, (
        "Validation loader should contain exactly 1 batch"
    )
    assert len(model.pytorch_loaders["test"]) == 1, (
        "Test loader should contain exactly 1 batch"
    )
    opts["trainer.n_validation_batches"] = None
    model = LuxonisModel(cfg, opts)
    assert len(model.pytorch_loaders["val"]) > 1, (
        "Validation loader should contain all validation samples"
    )
    assert len(model.pytorch_loaders["test"]) > 1, (
        "Test loader should contain all test samples"
    )
