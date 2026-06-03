from pathlib import Path
from typing import Any, Literal, cast

import pytest
from luxonis_ml.data import AlbumentationsEngine, LuxonisDataset
from luxonis_ml.data.loaders.luxonis_loader import LuxonisLoader
from luxonis_ml.typing import Params

from luxonis_train.config.config import (
    AugmentationConfig,
    Config,
    PreprocessingConfig,
)


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


def test_smart_cfg_auto_populate_validation_batch_limit():
    cfg = Config.get_config(
        cast(
            Params,
            {
                "model": {"nodes": [{"name": "ResNet"}]},
                "loader": {
                    "train_view": "train",
                    "val_view": "train",
                    "test_view": "train",
                },
            },
        )
    )
    assert cfg.trainer.n_validation_batches == 10

    cfg = Config.get_config(
        cast(
            Params,
            {
                "model": {"nodes": [{"name": "ResNet"}]},
                "loader": {
                    "train_view": "train",
                    "val_view": "train",
                    "test_view": "train",
                },
                "trainer": {"n_validation_batches": -1},
            },
        )
    )
    assert cfg.trainer.n_validation_batches == -1


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


def test_augmentation_apply_on_stages_defaults_to_train():
    cfg = Config.get_config(
        cast(
            Params,
            {
                "model": {"nodes": [{"name": "ResNet"}]},
                "trainer": {
                    "preprocessing": {"augmentations": [{"name": "Defocus"}]}
                },
            },
        )
    )

    assert cfg.trainer.preprocessing.augmentations[0].apply_on_stages == [
        "train"
    ]


def test_get_active_augmentations_preserves_apply_on_stages():
    cfg = Config.get_config(
        cast(
            Params,
            {
                "model": {"nodes": [{"name": "ResNet"}]},
                "trainer": {
                    "preprocessing": {
                        "augmentations": [
                            {
                                "name": "Defocus",
                                "apply_on_stages": ["train", "test"],
                            }
                        ]
                    }
                },
            },
        )
    )

    active_augmentations = cfg.trainer.preprocessing.get_active_augmentations()

    assert active_augmentations[0].apply_on_stages == ["train", "test"]


def test_apply_on_stages_reaches_stage_specific_augmentation_engines():
    def get_transform_names(wrapped_transform: Any) -> list[str]:
        if wrapped_transform is None:
            return []
        return next(
            (
                [type(item).__name__ for item in cell.cell_contents.transforms]
                for cell in wrapped_transform.__closure__
                if hasattr(cell.cell_contents, "transforms")
            ),
            [],
        )

    def resolve_pipeline_stage(
        view: list[str],
    ) -> Literal["train", "val", "test"]:
        loader = LuxonisLoader.__new__(LuxonisLoader)
        loader.view = view
        return cast(
            Literal["train", "val", "test"],
            loader._get_augmentation_pipeline_stage(),
        )

    preprocessing = PreprocessingConfig(
        augmentations=[
            AugmentationConfig(
                name="HorizontalFlip",
                params={"p": 1.0},
                apply_on_stages=["train"],
            ),
            AugmentationConfig(
                name="Defocus",
                params={"p": 1.0},
                apply_on_stages=["train", "val"],
            ),
        ]
    )
    engine_config = [
        augmentation.model_dump(exclude={"active"})
        for augmentation in preprocessing.get_active_augmentations()
    ]

    expected = {
        "train": {
            "spatial": ["HorizontalFlip"],
            "pixel": ["Defocus", "Normalize"],
        },
        "val": {
            "spatial": [],
            "pixel": ["Defocus", "Normalize"],
        },
        "test": {
            "spatial": [],
            "pixel": ["Normalize"],
        },
    }

    for loader_name, view in {
        "train": ["train"],
        "val": ["val"],
        "test": ["test"],
    }.items():
        engine = AlbumentationsEngine(
            256,
            256,
            {"/classification": "classification"},
            {"/classification": 1},
            ["image"],
            engine_config,
            pipeline_stage=resolve_pipeline_stage(view),
        )

        actual = {
            "spatial": get_transform_names(engine.spatial_transform),
            "pixel": get_transform_names(engine.pixel_transform),
        }

        assert actual == expected[loader_name]


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
    ArchiveOnTrainEnd.
    """
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
