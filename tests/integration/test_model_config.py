import os

from luxonis_ml.data import LuxonisDataset

from luxonis_train import LuxonisModel
from luxonis_train.config.config import Config as LuxonisTrainConfig


def test_config_dump(coco_dataset: LuxonisDataset):
    model_config_path = "configs/detection_light_model.yaml"
    temp_config_path = "tests/integration/test_saved_config.yaml"

    try:
        config = LuxonisTrainConfig().get_config(model_config_path)
        config.save_data(temp_config_path)

        opts = {
            "loader.params.dataset_name": coco_dataset.identifier,
        }

        model1 = LuxonisModel(cfg=temp_config_path, opts=opts)
        model2 = LuxonisModel(cfg=model_config_path, opts=opts)

        model1_config = model1.cfg.dict()
        model2_config = model2.cfg.dict()

        assert model1_config == model2_config, "Model configs are not the same"
        assert "Normalize" not in [
            aug["name"]
            for aug in model1_config["trainer"]["preprocessing"][
                "augmentations"
            ]
        ]
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def test_explicit_dataset_type():
    model_config_path = "configs/detection_light_model.yaml"
    temp_config_path = "tests/integration/test_saved_config.yaml"

    try:
        config1 = LuxonisTrainConfig().get_config(
            model_config_path,
            [
                "loader.params",
                {"dataset_name": "coco_test", "dataset_type": "coco"},
            ],
        )
        config1.save_data(temp_config_path)

        config2 = LuxonisTrainConfig().get_config(temp_config_path)
        assert (
            config1.loader.params["dataset_type"]
            == config2.loader.params["dataset_type"]
        )
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
