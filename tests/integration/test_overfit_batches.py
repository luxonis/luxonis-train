from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def test_overfit_batches_training(
    parking_lot_dataset: LuxonisDataset, opts: Params
):
    """Smoke test for overfit_batches passed."""
    cfg = "configs/detection_light_model.yaml"
    opts |= {
        "model.predefined_model.params.task_name": "vehicles",
        "loader.params.dataset_name": parking_lot_dataset.identifier,
        "loader.train_view": "train",
        "loader.val_view": "train",
        "loader.test_view": "train",
        "trainer.overfit_batches": 1,
        "trainer.seed": 42,
        "trainer.epochs": 1,
    }
    model = LuxonisModel(cfg, opts)

    assert model.cfg.trainer.overfit_batches == 1
    assert model.pl_trainer.overfit_batches == 1

    model.train()
