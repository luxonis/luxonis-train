from pathlib import Path

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train import LuxonisModel
from luxonis_train.config.config import Config as LuxonisTrainConfig


def test_config_dump(coco_dataset: LuxonisDataset):
    model_config_path = Path("configs", "detection_light_model.yaml")
    temp_config_path = Path("tests", "integration", "test_saved_config.yaml")

    try:
        config = LuxonisTrainConfig().get_config(str(model_config_path))
        config.save_data(temp_config_path)

        opts: Params = {
            "loader.params.dataset_name": coco_dataset.identifier,
        }

        model1 = LuxonisModel(cfg=str(temp_config_path), opts=opts)
        model2 = LuxonisModel(cfg=str(model_config_path), opts=opts)

        model1_config = model1.cfg.model_dump()
        model2_config = model2.cfg.model_dump()

        assert model1_config == model2_config, "Model configs are not the same"
        assert "Normalize" not in [
            aug["name"]
            for aug in model1_config["trainer"]["preprocessing"][
                "augmentations"
            ]
        ]
    finally:
        temp_config_path.unlink(missing_ok=True)
