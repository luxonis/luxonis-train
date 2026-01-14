from pathlib import Path
from typing import cast

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.config.config import Config


def test_smart_cfg_auto_populate(coco_dataset: LuxonisDataset):
    cfg = {
        "model": {
            "name": "test_auto_populate",
            "predefined_model": {
                "name": "DetectionModel",
                "params": {
                    "loss_params": {
                        "iou_type": "siou",
                        "iout_loss_weight": 14,
                        "class_loss_weight": 1,
                    }
                },
            },
        },
        "trainer": {
            "batch_size": 2,
            "epochs": 10,
            "scheduler": {"name": "CosineAnnealingLR"},
            "preprocessing": {
                "train_image_size": [128, 128],
                "augmentations": [{"name": "Mosaic4"}],
            },
        },
        "loader": {"params": {"dataset_name": coco_dataset.identifier}},
    }

    cfg = Config.get_config(cfg)

    assert cfg.trainer.scheduler is not None
    scheduler_params = cfg.trainer.scheduler.params
    assert scheduler_params["T_max"] == cfg.trainer.epochs

    augmentations = cfg.trainer.preprocessing.augmentations[0].params
    img_width, img_height = cfg.trainer.preprocessing.train_image_size
    assert augmentations["out_width"] == img_width
    assert augmentations["out_height"] == img_height

    batch_size = cfg.trainer.batch_size
    grad_accumulation = 64 // batch_size

    assert cfg.model.predefined_model is not None
    loss_params = cfg.model.predefined_model.params["loss_params"]
    expected_iou_weight = 2.5 * grad_accumulation
    expected_class_weight = 1.0 * grad_accumulation
    assert isinstance(loss_params, dict)

    assert loss_params["iou_loss_weight"] == expected_iou_weight
    assert loss_params["class_loss_weight"] == expected_class_weight


def test_config_dump(coco_dataset: LuxonisDataset, tmp_path: Path):
    model_config_path = Path("configs", "detection_light_model.yaml")
    temp_config_path = tmp_path / "config.yaml"

    config = Config.get_config(model_config_path)
    config.save_data(temp_config_path)

    opts: Params = {
        "loader.params.dataset_name": coco_dataset.identifier,
    }

    cfg1 = Config.get_config(temp_config_path, opts).model_dump()
    cfg2 = Config.get_config(model_config_path, opts).model_dump()

    assert cfg1 == cfg2, "Model configs are not the same"
    assert "Normalize" not in [
        aug["name"]
        for aug in cfg1["trainer"]["preprocessing"]["augmentations"]
    ]


def test_explicit_dataset_type(tmp_path: Path):
    model_config_path = Path("configs", "detection_light_model.yaml")
    temp_config_path = tmp_path / "config.yaml"

    cfg1 = Config.get_config(
        model_config_path,
        {
            "loader.params": {
                "dataset_name": "coco_test",
                "dataset_type": "coco",
            },
        },
    )
    cfg1.save_data(temp_config_path)

    cfg2 = Config.get_config(temp_config_path)
    assert (
        cfg1.loader.params["dataset_type"]
        == cfg2.loader.params["dataset_type"]
    )


def test_config_invalid():
    cfg: Params = {
        "model": {
            "nodes": [
                {"name": "ResNet"},
                {
                    "name": "SegmentationHead",
                    "metrics": [
                        {
                            "name": "Accuracy",
                            "is_main_metric": True,
                        },
                        {
                            "name": "JaccardIndex",
                            "is_main_metric": True,
                        },
                    ],
                },
            ]
        },
    }
    with pytest.raises(ValueError, match="Only one main metric"):
        Config.get_config(cfg)


@pytest.mark.parametrize(
    ("callbacks_input", "expected_active"),
    [
        (
            [{"name": "ConvertOnTrainEnd"}],
            {"ConvertOnTrainEnd": True},
        ),
        (
            [{"name": "ExportOnTrainEnd"}],
            {"ExportOnTrainEnd": True},
        ),
        (
            [{"name": "ArchiveOnTrainEnd"}],
            {"ArchiveOnTrainEnd": True},
        ),
        (
            [{"name": "ExportOnTrainEnd"}, {"name": "ArchiveOnTrainEnd"}],
            {"ExportOnTrainEnd": True, "ArchiveOnTrainEnd": True},
        ),
        (
            [{"name": "ConvertOnTrainEnd"}, {"name": "ExportOnTrainEnd"}],
            {"ConvertOnTrainEnd": True, "ExportOnTrainEnd": False},
        ),
        (
            [{"name": "ConvertOnTrainEnd"}, {"name": "ArchiveOnTrainEnd"}],
            {"ConvertOnTrainEnd": True, "ArchiveOnTrainEnd": False},
        ),
        (
            [
                {"name": "ConvertOnTrainEnd"},
                {"name": "ExportOnTrainEnd"},
                {"name": "ArchiveOnTrainEnd"},
            ],
            {
                "ConvertOnTrainEnd": True,
                "ExportOnTrainEnd": False,
                "ArchiveOnTrainEnd": False,
            },
        ),
        (
            [
                {"name": "ConvertOnTrainEnd"},
                {"name": "ExportOnTrainEnd", "active": False},
            ],
            {"ConvertOnTrainEnd": True, "ExportOnTrainEnd": False},
        ),
    ],
)
def test_convert_callback_deactivates_export_and_archive(
    callbacks_input: list[Params], expected_active: dict[str, bool]
):
    """Test that ConvertOnTrainEnd deactivates ExportOnTrainEnd and
    ArchiveOnTrainEnd."""
    cfg = Config.get_config(
        cast(
            Params,
            {
                "model": {"nodes": [{"name": "ResNet"}]},
                "trainer": {"callbacks": callbacks_input},
            },
        )
    )

    callbacks_by_name = {cb.name: cb for cb in cfg.trainer.callbacks}

    for callback_name, should_be_active in expected_active.items():
        assert callback_name in callbacks_by_name, (
            f"Callback '{callback_name}' not found in config"
        )
        assert callbacks_by_name[callback_name].active == should_be_active, (
            f"Callback '{callback_name}' expected active={should_be_active}, "
            f"got active={callbacks_by_name[callback_name].active}"
        )
