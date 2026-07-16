import torch
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params
from torch import Size, Tensor
from typing_extensions import override

from luxonis_train.core import LuxonisModel
from luxonis_train.core.utils.aimet_utils import get_ptq_calibration_loader
from luxonis_train.loaders import BaseLoaderTorch
from luxonis_train.typing import Labels


class SplitLengthLoader(BaseLoaderTorch):
    LENGTHS = {
        ("train",): 20,
        ("val",): 3,
        ("test",): 11,
        ("train", "val", "test"): 34,
    }

    @property
    @override
    def input_shapes(self) -> dict[str, Size]:
        return {self.image_source: Size([3, self.height, self.width])}

    @override
    def __len__(self) -> int:
        return self.LENGTHS[tuple(self.view)]

    @override
    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor | dict[str, Tensor], Labels]:
        image = torch.zeros(3, self.height, self.width)
        labels = {"vehicles/boundingbox": torch.zeros(1, 5)}
        return image, labels

    @override
    def get_classes(self) -> dict[str, dict[str, int]]:
        return {"vehicles": {"car": 0}}


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


def test_unlimited_validation_batches_keep_full_eval_loaders():
    model = LuxonisModel(
        {
            "model": {
                "predefined_model": {
                    "name": "DetectionModel",
                    "params": {"task_name": "vehicles"},
                }
            },
            "loader": {
                "name": "SplitLengthLoader",
                "train_view": "train",
                "val_view": "val",
                "test_view": ["train", "val", "test"],
            },
            "trainer": {
                "batch_size": 1,
                "n_validation_batches": -1,
            },
        }
    )

    assert len(model.loaders["val"]) == 3
    assert len(model.loaders["test"]) == 34
    assert model.cfg.trainer.n_validation_batches == -1
    assert len(model.pytorch_loaders["val"]) == len(model.loaders["val"])
    assert len(model.pytorch_loaders["test"]) == len(model.loaders["test"])


def test_ptq_calibration_loader_is_independent_from_validation_batch_limit():
    model = LuxonisModel(
        {
            "model": {
                "predefined_model": {
                    "name": "DetectionModel",
                    "params": {"task_name": "vehicles"},
                }
            },
            "loader": {
                "name": "SplitLengthLoader",
                "train_view": "train",
                "val_view": "val",
                "test_view": "test",
            },
            "trainer": {
                "batch_size": 1,
                "n_validation_batches": 1,
            },
            "exporter": {"aimet": {"max_calibration_images": 2}},
        }
    )

    assert len(model.pytorch_loaders["val"]) == 1
    assert (
        len(
            get_ptq_calibration_loader(
                val_dataset=model.loaders["val"],
                collate_fn=model.loaders["val"].collate_fn,
                batch_size=model.cfg.trainer.batch_size,
                num_workers=model.cfg.trainer.n_workers,
                pin_memory=model.cfg.trainer.pin_memory,
                max_calibration_images=None,
            )
        )
        == 3
    )
    assert (
        len(
            get_ptq_calibration_loader(
                val_dataset=model.loaders["val"],
                collate_fn=model.loaders["val"].collate_fn,
                batch_size=model.cfg.trainer.batch_size,
                num_workers=model.cfg.trainer.n_workers,
                pin_memory=model.cfg.trainer.pin_memory,
                max_calibration_images=2,
            )
        )
        == 2
    )
