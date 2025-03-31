from luxonis_ml.data import LuxonisDataset

from luxonis_train import LuxonisModel


def test_loader_splits(coco_dataset: LuxonisDataset):
    cfg_path = "tests/configs/config_simple.yaml"

    tests = [
        (("train", "val", "test"), {"train": 24, "val": 3, "test": 3}),
        (("train", "train", "train"), {"train": 24, "val": 24, "test": 24}),
        (("test", "test", "test"), {"train": 3, "val": 3, "test": 3}),
        (("val", "val", "val"), {"train": 3, "val": 3, "test": 3}),
    ]

    for (train_view, val_view, test_view), expected in tests:
        opts = {
            "loader.params.dataset_name": coco_dataset.identifier,
            "trainer.batch_size": 1,
        }
        opts["loader.train_view"] = train_view
        opts["loader.val_view"] = val_view
        opts["loader.test_view"] = test_view
        opts["trainer.smart_cfg_auto_populate"] = False

        model = LuxonisModel(cfg=cfg_path, opts=opts)

        for key, exp_len in expected.items():
            assert len(model.pytorch_loaders[key]) == exp_len
