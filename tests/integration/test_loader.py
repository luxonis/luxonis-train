from typing import Literal

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train import LuxonisModel
from luxonis_train.core import LuxonisModel


@pytest.mark.parametrize(
    ("train_view", "val_view", "test_view", "expected"),
    [
        ("train", "val", "test", {"train": 24, "val": 3, "test": 3}),
        ("train", "train", "train", {"train": 24, "val": 24, "test": 24}),
        ("test", "test", "test", {"train": 3, "val": 3, "test": 3}),
        ("val", "val", "val", {"train": 3, "val": 3, "test": 3}),
        (
            ["train", "val", "test"],
            ["val", "test"],
            ["train", "test"],
            {"train": 30, "val": 6, "test": 27},
        ),
    ],
)
def test_loader_splits(
    coco_dataset: LuxonisDataset,
    train_view: str | list[str],
    val_view: str | list[str],
    test_view: str | list[str],
    expected: dict[Literal["train", "val", "test"], int],
    opts: Params,
):
    cfg_path = "configs/detection_light_model.yaml"

    opts |= {
        "loader.params.dataset_name": coco_dataset.identifier,
    }
    opts["loader.train_view"] = train_view
    opts["loader.val_view"] = val_view
    opts["loader.test_view"] = test_view
    opts["trainer.smart_cfg_auto_populate"] = False

    model = LuxonisModel(cfg=cfg_path, opts=opts)

    for key, exp_len in expected.items():
        assert len(model.pytorch_loaders[key]) == exp_len


@pytest.mark.parametrize(
    ("train_view", "val_view", "test_view", "expected"),
    [
        ("train", "val", "test", {"train": 24, "val": 3, "test": 3}),
        ("train", "train", "train", {"train": 24, "val": 24, "test": 24}),
        ("test", "test", "test", {"train": 3, "val": 3, "test": 3}),
        ("val", "val", "val", {"train": 3, "val": 3, "test": 3}),
        (
            ["train", "val", "test"],
            "val",
            "test",
            {"train": 30, "val": 3, "test": 3},
        ),
    ],
)
def test_splits(
    coco_dataset: LuxonisDataset,
    train_view: str | list[str],
    val_view: str,
    test_view: str,
    expected: dict[Literal["train", "val", "test"], int],
    opts: Params,
):
    cfg = "configs/detection_light_model.yaml"

    opts |= {
        "trainer.batch_size": 1,
        "trainer.smart_cfg_auto_populate": False,
        "loader.params.dataset_name": coco_dataset.identifier,
        "loader.train_view": train_view,
        "loader.val_view": val_view,
        "loader.test_view": test_view,
    }

    model = LuxonisModel(cfg, opts)

    for key, exp_len in expected.items():
        assert len(model.pytorch_loaders[key]) == exp_len


def test_parsing(opts: Params):
    opts |= {
        "loader.params.dataset_name": "parsed_coco_test",
        "loader.params.dataset_dir": (
            "gs://luxonis-test-bucket/luxonis-train-test-data"
            "/datasets/COCO_people_subset.zip"
        ),
    }
    model = LuxonisModel("configs/detection_light_model.yaml", opts)
    model.train()
