import os

from luxonis_ml.data import LuxonisDataset

from luxonis_train import LuxonisModel
from luxonis_train.config.config import Config as LuxonisTrainConfig


def normalize_dict(d):
    """Convert any nested dictionaries to sorted items for
    comparison."""
    if isinstance(d, dict):
        return {k: normalize_dict(v) for k, v in sorted(d.items())}
    elif isinstance(d, list):
        return [normalize_dict(x) for x in sorted(d, key=str) if x is not None]
    else:
        return d


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

        model1_config = normalize_dict(model1.cfg.model_dump())
        model2_config = normalize_dict(model2.cfg.model_dump())

        assert model1_config == model2_config, "Model configs are not the same"
        assert "Noramlize" not in [
            aug["name"]
            for aug in model1_config["trainer"]["preprocessing"][
                "augmentations"
            ]
        ]
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
