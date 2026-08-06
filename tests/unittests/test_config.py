import sys
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import patch

import pytest
from loguru import logger
from luxonis_ml.typing import Params
from pydantic import ValidationError

from luxonis_train.config import (
    AttachedModuleConfig,
    BasePredefinedModel,
    Config,
    ExportConfig,
    LossModuleConfig,
    MetricModuleConfig,
    NodeConfig,
    TrainerConfig,
    predefined,
)
from luxonis_train.config.config import (
    AIMETConfig,
    ArchiveConfig,
    AugmentationConfig,
    BlobconverterExportConfig,
    FinetuningConfig,
    FreezingConfig,
    HubAIExportConfig,
    LoaderConfig,
    ModelConfig,
    NormalizeAugmentationConfig,
    OnnxExportConfig,
    OptimizerConfig,
    ParameterPattern,
    PredefinedModelConfig,
    PreprocessingConfig,
    SchedulerConfig,
    StorageConfig,
    TrackerConfig,
    TunerConfig,
    _validate_quantization_mode,
)
from luxonis_train.config.predefined import (
    VARIANT_ORDER,
    _model_class,
    configs_dir,
    list_predefined_models,
    list_variants,
    resolve_predefined_config,
)
from luxonis_train.config.predefined_models import (
    AnomalyDetectionModel,
    ClassificationModel,
    DetectionModel,
    EmbeddingsModel,
    FOMOModel,
    InstanceSegmentationModel,
    KeypointDetectionModel,
    OCRRecognitionModel,
    SegmentationModel,
)
from luxonis_train.config.predefined_models.base_predefined_model import (
    SimplePredefinedModel,
)
from luxonis_train.config.predefined_versions import (
    _split_family_version,
    list_versions,
    resolve_predefined_class,
    resolved_class_name,
    warn_on_predefined_model_mismatch,
)
from luxonis_train.registry import MODELS

BASE_MODEL_CFG: Params = {
    "model": {"nodes": [{"name": "Backbone"}]},
    "trainer": {"smart_cfg_auto_populate": False},
}


class ConcreteSimplePredefinedModel(SimplePredefinedModel):
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "default", {"default": {}}


@pytest.mark.parametrize(
    "path", sorted(configs_dir().glob("*.yaml")), ids=lambda path: path.name
)
def test_config_load(path: Path):
    cfg = Config.get_config(path)

    assert cfg.model is not None
    assert cfg.trainer is not None
    assert cfg.loader is not None


def test_complex_model_conv2d_finetuning_uses_module_type_selector():
    cfg = Config.get_config(configs_dir() / "complex_model.yaml")
    node = next(
        node
        for node in cfg.model.nodes
        if node.name == "EfficientKeypointBBoxHead"
    )

    parameters = node.finetuning[1].parameters
    assert parameters is not None
    pattern = parameters[0]

    assert pattern.name is None
    assert pattern.module_type == "Conv2d"


def test_parameter_pattern_helpers_and_validation():
    with pytest.raises(ValueError, match="At least one"):
        ParameterPattern()
    with pytest.raises(ValueError, match=r"name.*empty"):
        ParameterPattern(name="")
    with pytest.raises(ValueError, match=r"module_type.*empty"):
        ParameterPattern(module_type="")

    pattern = ParameterPattern(name="fc", module_type="Linear")
    assert pattern.matches("Linear", "classifier.fc.weight") is True
    assert pattern.matches("Linear", "classifier.conv.weight") is False
    assert pattern.matches("Conv2d", "classifier.fc.weight") is False


def test_optimizer_scheduler_and_finetuning_config_helpers():
    scheduler = SchedulerConfig(
        name="SequentialLR",
        params={
            "schedulers": [{"name": "LinearLR", "params": {"total_iters": 2}}],
            "milestones": [1],
        },
    )
    sequential_params = scheduler.get_sequential_lr_params()

    assert sequential_params.schedulers[0].name == "LinearLR"
    assert sequential_params.milestones == [1]
    assert scheduler.to_finetuning().name == "SequentialLR"
    assert OptimizerConfig(name="SGD").to_finetuning().name == "SGD"

    with pytest.raises(RuntimeError, match="not 'SequentialLR'"):
        SchedulerConfig(name="StepLR").get_sequential_lr_params()
    with pytest.raises(ValueError, match="requires 'schedulers'"):
        SchedulerConfig(name="SequentialLR").get_sequential_lr_params()


def test_finetuning_config_parameter_normalization():
    existing = ParameterPattern(module_type="Linear")
    cfg = FinetuningConfig(
        parameters=cast(Any, ["fc", {"name": "head"}, existing])
    )

    assert cfg.parameters is not None
    assert [(p.name, p.module_type) for p in cfg.parameters] == [
        ("fc", None),
        ("head", None),
        (None, "Linear"),
    ]

    with pytest.raises(ValidationError):
        FinetuningConfig(parameters=cast(Any, 1))
    with pytest.raises(ValueError, match="at least one parameter pattern"):
        FinetuningConfig(parameters=[])
    with pytest.raises(TypeError, match="Parameter patterns must be"):
        FinetuningConfig(parameters=cast(Any, [object()]))


def test_public_config_exports_are_importable():
    assert AttachedModuleConfig(name="Metric", alias="alias").identifier == (
        "alias"
    )
    assert BasePredefinedModel.__name__ == "BasePredefinedModel"
    assert ExportConfig().quantization_mode == "INT8_STANDARD"
    assert LossModuleConfig(name="loss", weight=0).identifier == "loss"
    assert MetricModuleConfig(name="metric").is_main_metric is False
    assert NodeConfig(name="node", alias="alias").identifier == "alias"
    assert TrainerConfig().optimizer.name == "Adam"
    assert ArchiveConfig().upload_to_run is True
    assert BlobconverterExportConfig().version == "2022.1"
    assert OnnxExportConfig().opset_version == 16
    assert TrackerConfig().is_tensorboard is True
    assert TunerConfig().storage.active is True
    assert StorageConfig(active=False).active is False
    assert NormalizeAugmentationConfig().active is True
    assert PredefinedModelConfig(name="DetectionModel").variant == "default"
    assert FreezingConfig(active=True).active is True


def test_config_dump_roundtrip_without_dataset_fixture(tmp_path: Path):
    model_config_path = Path(
        "luxonis_train", "configs", "detection_light_model.yaml"
    )
    temp_config_path = tmp_path / "config.yaml"

    cfg = Config.get_config(
        model_config_path,
        {"loader.params": {"dataset_name": "coco_test"}},
    )
    cfg.save_data(temp_config_path)

    cfg1 = Config.get_config(temp_config_path).model_dump()
    cfg2 = Config.get_config(
        model_config_path,
        {"loader.params": {"dataset_name": "coco_test"}},
    ).model_dump()

    assert cfg1 == cfg2
    assert "ENVIRON" not in cfg.model_dump()
    assert "ENVIRON" not in cfg.model_dump_json()
    assert "Normalize" not in [
        aug["name"]
        for aug in cfg1["trainer"]["preprocessing"]["augmentations"]
    ]


def test_model_node_validation_linearizes_heads_and_outputs():
    model = ModelConfig.model_validate(
        {
            "nodes": [
                {"name": "Backbone"},
                {"name": "Neck"},
                {"name": "FirstHead"},
                {"name": "SecondHead"},
            ]
        }
    )

    assert [node.inputs for node in model.nodes] == [
        [],
        ["Backbone"],
        ["Neck"],
        ["Neck"],
    ]
    assert model.outputs == ["FirstHead", "SecondHead"]


def test_model_node_validation_rejects_missing_name():
    with pytest.raises(ValueError, match="does not specify"):
        ModelConfig.model_validate({"nodes": [{}]})


def test_model_config_accepts_existing_node_instances():
    node = NodeConfig(name="Backbone")
    assert ModelConfig(nodes=[node]).nodes == [node]


def test_model_config_rejects_invalid_graph_and_names():
    with pytest.raises(ValueError, match="not acyclic"):
        ModelConfig.model_validate(
            {
                "nodes": [
                    {"name": "A", "inputs": ["B"]},
                    {"name": "B", "inputs": ["A"]},
                ]
            }
        )

    with pytest.raises(ValueError, match="contain a '/'"):
        ModelConfig.model_validate({"nodes": [{"name": "Invalid/Node"}]})

    with pytest.raises(ValueError, match="contain a '/'"):
        ModelConfig.model_validate(
            {
                "nodes": [
                    {
                        "name": "Head",
                        "metrics": [{"name": "Invalid/Metric"}],
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="contain a '/'"):
        ModelConfig.model_validate(
            {
                "nodes": [
                    {
                        "name": "Head",
                        "metrics": [
                            {"name": "Metric", "alias": "Invalid/Alias"}
                        ],
                    }
                ]
            }
        )


def test_model_config_no_outputs_guard(monkeypatch: pytest.MonkeyPatch):
    import luxonis_train.config.config as config_module

    monkeypatch.setattr(config_module, "is_acyclic", lambda graph: True)
    model = ModelConfig.model_construct(
        nodes=[NodeConfig(name="A", inputs=["A"])],
        outputs=[],
    )

    with pytest.raises(ValueError, match="No outputs"):
        cast(Any, model).check_graph()


def test_model_config_main_metric_and_duplicate_name_handling():
    model = ModelConfig.model_validate(
        {
            "nodes": [
                {
                    "name": "Head",
                    "alias": "dup",
                    "losses": [
                        {"name": "dup"},
                        {"name": "Other", "alias": "dup"},
                    ],
                    "metrics": [
                        {"name": "Accuracy"},
                        {"name": "Accuracy"},
                    ],
                }
            ]
        }
    )

    node = model.nodes[0]
    assert node.losses[0].alias == "dup_dup"
    assert node.losses[1].alias == "dup_0"
    assert node.metrics[0].is_main_metric is True
    assert node.metrics[1].alias == "Accuracy_dup"

    nested_node = NodeConfig.model_construct(name="dup", alias=None)
    node_with_node_module = NodeConfig.model_construct(
        name="dup",
        alias=None,
        losses=[nested_node],
        metrics=[],
        visualizers=[],
    )
    model = ModelConfig.model_construct(
        nodes=[node_with_node_module], outputs=["dup"]
    )
    cast(Any, model).check_unique_names()
    assert nested_node.alias == "dup_0"

    with pytest.raises(ValueError, match="Only one main metric"):
        ModelConfig.model_validate(
            {
                "nodes": [
                    {
                        "name": "Head",
                        "metrics": [
                            {"name": "Accuracy", "is_main_metric": True},
                            {"name": "JaccardIndex", "is_main_metric": True},
                        ],
                    }
                ]
            }
        )


def test_head_nodes_uses_lazy_nodes_import(monkeypatch: pytest.MonkeyPatch):
    class FakeBaseHead:
        pass

    class FakeOutput(FakeBaseHead):
        pass

    fake_nodes = ModuleType("luxonis_train.nodes")
    cast(Any, fake_nodes).BaseHead = FakeBaseHead
    monkeypatch.setitem(sys.modules, "luxonis_train.nodes", fake_nodes)

    import luxonis_train.config.config as config_module

    monkeypatch.setitem(
        config_module.NODES._module_dict, "FakeOutput", FakeOutput
    )

    model = ModelConfig.model_validate(
        {
            "nodes": [
                {"name": "FakeOutput"},
                {"name": "OtherNode"},
            ]
        }
    )

    assert [node.name for node in model.head_nodes] == ["FakeOutput"]


def test_loader_config_validation_and_serialization():
    cfg = LoaderConfig.model_validate(
        {
            "name": "DummyLoader",
            "train_view": "train",
            "val_view": ["val"],
            "test_view": "test",
            "params": {
                "dataset_type": "coco",
                "n_classes": 3,
                "n_keypoints": 2,
                "class_names": ["a", "b", "c"],
                "kept": True,
            },
        }
    )

    dumped = cfg.model_dump()
    assert dumped["name"] == "LuxonisLoaderTorch"
    assert dumped["params"] == {"dataset_type": "coco", "kept": True}
    assert cfg.train_view == ["train"]
    assert cfg.test_view == ["test"]

    with pytest.raises(TypeError, match="dataset_type"):
        LoaderConfig(params={"dataset_type": 1})

    with pytest.raises(ValueError, match="not supported"):
        LoaderConfig(params={"dataset_type": "unknown"})

    with pytest.raises(TypeError, match="train_view"):
        LoaderConfig.model_validate({"train_view": 1})


def test_preprocessing_normalization_and_resizing():
    cfg = PreprocessingConfig.model_validate(
        {
            "train_image_size": (128, 256),
            "augmentations": [
                AugmentationConfig(name="Normalize", params={"mean": [0]}),
                AugmentationConfig(
                    name="Resize",
                    params={"height": 1, "width": 2},
                    use_for_resizing=True,
                ),
                AugmentationConfig(name="Inactive", active=False),
                AugmentationConfig(
                    name="Active", apply_on_stages=["train", "val"]
                ),
            ],
        }
    )

    assert cfg.normalize.params == {"mean": [0]}
    resize = next(aug for aug in cfg.augmentations if aug.name == "Resize")
    assert resize.params == {"height": 128, "width": 256, "p": 1.0}
    assert [aug.name for aug in cfg.get_active_augmentations()] == [
        "Resize",
        "Active",
        "Normalize",
    ]
    assert cfg.get_active_augmentations()[1].apply_on_stages == [
        "train",
        "val",
    ]
    assert "Normalize" not in [
        aug["name"] for aug in cfg.model_dump()["augmentations"]
    ]

    assert (
        PreprocessingConfig(
            normalize=NormalizeAugmentationConfig(active=False)
        ).augmentations
        == []
    )


@pytest.mark.parametrize(
    ("trainer", "expected"),
    [
        (
            TrainerConfig.model_validate(
                {
                    "epochs": 7,
                    "scheduler": {"name": "CosineAnnealingLR"},
                }
            ),
            {},
        ),
        (
            TrainerConfig.model_validate(
                {
                    "epochs": 7,
                    "scheduler": {
                        "name": "CosineAnnealingLR",
                        "params": {"T_max": 3},
                    },
                }
            ),
            {"T_max": 3},
        ),
    ],
)
def test_trainer_scheduler_validation(
    trainer: TrainerConfig, expected: dict[str, int]
):
    assert trainer.scheduler.params == expected


def test_trainer_validation_branches(monkeypatch: pytest.MonkeyPatch):
    trainer = TrainerConfig.model_validate(
        {
            "seed": 1,
            "deterministic": None,
            "overfit_batches": 1,
            "validation_interval": 10,
            "epochs": 2,
            "callbacks": [
                {"name": "Later"},
                {"name": "EMACallback"},
                {
                    "name": "GradientAccumulationScheduler",
                    "params": {"scheduling": {"1": 2, "x": 3}},
                },
            ],
        }
    )

    assert trainer.deterministic is True
    assert trainer.validation_interval == 2
    assert [callback.name for callback in trainer.callbacks[:2]] == [
        "EMACallback",
        "Later",
    ]
    assert trainer.callbacks[2].params["scheduling"] == {1: 2, "x": 3}

    trainer = TrainerConfig.model_validate(
        {
            "callbacks": [
                {
                    "name": "GradientAccumulationScheduler",
                    "params": {"scheduling": ["not", "mapping"]},
                }
            ]
        }
    )
    assert trainer.callbacks[0].params["scheduling"] == ["not", "mapping"]

    monkeypatch.setattr(sys, "platform", "win32")
    assert TrainerConfig(n_workers=4).n_workers == 0


@pytest.mark.parametrize(
    ("callbacks_input", "expected_active"),
    [
        ([{"name": "ConvertOnTrainEnd"}], {"ConvertOnTrainEnd": True}),
        ([{"name": "ExportOnTrainEnd"}], {"ExportOnTrainEnd": True}),
        ([{"name": "ArchiveOnTrainEnd"}], {"ArchiveOnTrainEnd": True}),
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
    cfg = Config.get_config(
        cast(
            Params,
            BASE_MODEL_CFG
            | {
                "trainer": {
                    "smart_cfg_auto_populate": False,
                    "callbacks": callbacks_input,
                }
            },
        )
    )

    callbacks_by_name = {cb.name: cb for cb in cfg.trainer.callbacks}

    for callback_name, should_be_active in expected_active.items():
        assert callbacks_by_name[callback_name].active == should_be_active


def test_export_config_validation():
    assert _validate_quantization_mode("fp16") == "FP16_STANDARD"
    assert (
        ExportConfig.model_validate({"data_type": "fp32"}).quantization_mode
        == "FP32_STANDARD"
    )
    assert ExportConfig.model_validate(
        {"scale_values": 1, "mean_values": [2, 3, 4]}
    ).scale_values == [
        1,
        1,
        1,
    ]
    assert HubAIExportConfig(active=True, platform="rvc4").platform == "rvc4"

    with pytest.raises(ValueError, match="Invalid quantization_mode"):
        ExportConfig(quantization_mode="bad")

    with pytest.raises(ValueError, match="platform"):
        HubAIExportConfig(active=True)

    with pytest.raises(NotImplementedError, match="Hailo"):
        HubAIExportConfig(platform="hailo")


def test_aimet_config_path_loading_and_serialization(
    monkeypatch: pytest.MonkeyPatch,
):
    import luxonis_train.config.config as config_module

    class FakeFileSystem:
        def __init__(self, path: str):
            self.path = path

        def read_text(self, _: str) -> str:
            assert self.path == "memory://aimet.json"
            return '{"ops": {"is_output_quantized": "True"}}'

    monkeypatch.setattr(config_module, "LuxonisFileSystem", FakeFileSystem)

    cfg = ExportConfig.model_validate(
        {"aimet": {"config": "memory://aimet.json"}}
    )
    assert cfg.aimet.config == {"ops": {"is_output_quantized": "True"}}

    class FakeEnum(Enum):
        VALUE = "value"

    aimet = AIMETConfig.model_construct(
        default_data_type=FakeEnum.VALUE, quant_scheme=FakeEnum.VALUE
    )
    assert aimet.model_dump()["default_data_type"] == "VALUE"
    assert aimet.model_dump()["quant_scheme"] == "VALUE"


def test_aimet_config_path_loading_failure(monkeypatch: pytest.MonkeyPatch):
    import luxonis_train.config.config as config_module

    class FailingFileSystem:
        def __init__(self, path: str):
            self.path = path

        def read_text(self, _: str) -> str:
            raise RuntimeError("boom")

    monkeypatch.setattr(config_module, "LuxonisFileSystem", FailingFileSystem)

    with pytest.raises(ValueError, match="Failed to load AIMET config"):
        ExportConfig.model_validate({"aimet": {"config": "memory://bad.json"}})


def test_config_validators_and_storage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POSTGRES_USER", "user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pass")
    monkeypatch.setenv("POSTGRES_HOST", "host")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "db")

    cfg = Config.get_config(
        {
            "ENVIRON": {},
            "model": {"nodes": [{"name": "Backbone"}]},
            "trainer": {"smart_cfg_auto_populate": False},
            "tuner": {"storage": {"backend": "postgresql"}},
        }
    )

    assert cfg.tuner.storage.username == "user"
    assert cfg.tuner.storage.password is not None
    assert cfg.tuner.storage.password.get_secret_value() == "pass"
    assert cfg.tuner.storage.host == "host"
    assert cfg.tuner.storage.port == 5432
    assert cfg.tuner.storage.database == "db"

    assert Config.get_config(BASE_MODEL_CFG).tuner.storage.database == (
        "study_local.db"
    )

    with pytest.raises(TypeError, match="rich_logging"):
        Config.get_config({"rich_logging": "yes"})

    fake_utils = ModuleType("luxonis_train.utils")
    calls: list[bool] = []

    def setup_logging(*, use_rich: bool) -> None:
        calls.append(use_rich)

    cast(Any, fake_utils).setup_logging = setup_logging
    monkeypatch.setitem(sys.modules, "luxonis_train.utils", fake_utils)
    Config.get_config(BASE_MODEL_CFG | {"rich_logging": False})
    assert calls == [False]

    constructed = Config.model_construct(tuner=None)
    assert cast(Any, constructed).check_tune_storage() is constructed


def test_config_get_config_handles_string_mlflow_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    import luxonis_train.config.config as config_module

    class FakeFileSystem:
        def __init__(self, path: str):
            self.path = path
            self.is_mlflow = True
            self.experiment_id = "experiment"
            self.run_id = "run"

        @staticmethod
        def download(path: str, cache: Path) -> dict[str, Any]:
            return {
                "model": {"nodes": [{"name": "Backbone"}]},
                "trainer": {"smart_cfg_auto_populate": False},
            }

    monkeypatch.setattr(config_module, "LuxonisFileSystem", FakeFileSystem)

    cfg = Config.get_config("mlflow://config.yaml")

    assert cfg.tracker.project_id == "experiment"
    assert cfg.tracker.run_id == "run"


def test_smart_cfg_auto_populate_without_dataset_fixture():
    cfg = Config.get_config(
        {
            "model": {
                "name": "test_auto_populate",
                "predefined_model": {
                    "name": "DetectionModel",
                    "params": {
                        "loss_params": {
                            "iou_type": "siou",
                            "iou_loss_weight": 14,
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
            "loader": {"params": {"dataset_name": "coco_test"}},
        }
    )

    assert cfg.trainer.scheduler.params == {}
    assert (
        cfg.trainer.preprocessing.augmentations[0].params["out_width"] == 128
    )
    assert (
        cfg.trainer.preprocessing.augmentations[0].params["out_height"] == 128
    )
    assert cfg.trainer.accumulate_grad_batches == 32
    assert cfg.model.predefined_model is not None
    loss_params = cast(
        dict[str, Any], cfg.model.predefined_model.params["loss_params"]
    )
    assert loss_params["iou_loss_weight"] == 80.0
    assert loss_params["class_loss_weight"] == 32
    assert [cb.name for cb in cfg.trainer.callbacks] == [
        "UploadCheckpoint",
        "TestOnTrainEnd",
        "ConvertOnTrainEnd",
    ]


def test_smart_cfg_auto_populate_validation_batch_limit():
    cfg = Config.get_config(
        {
            "model": {"nodes": [{"name": "Backbone"}]},
            "loader": {
                "train_view": "train",
                "val_view": "train",
                "test_view": "train",
            },
        }
    )
    assert cfg.trainer.n_validation_batches == 10

    cfg = Config.get_config(
        {
            "model": {"nodes": [{"name": "Backbone"}]},
            "loader": {
                "train_view": "train",
                "val_view": "train",
                "test_view": "train",
            },
            "trainer": {"n_validation_batches": -1},
        }
    )
    assert cfg.trainer.n_validation_batches == -1


@pytest.mark.parametrize(
    ("model_name", "expected_weights"),
    [
        (
            "InstanceSegmentationModel",
            {
                "bbox_loss_weight": 240.0,
                "class_loss_weight": 16.0,
                "dfl_loss_weight": 48.0,
            },
        ),
        (
            "KeypointDetectionModel",
            {
                "iou_loss_weight": 240.0,
                "class_loss_weight": 16.0,
                "regr_kpts_loss_weight": 384,
                "vis_kpts_loss_weight": 32,
            },
        ),
    ],
)
def test_smart_cfg_auto_populate_predefined_schedules(
    model_name: str, expected_weights: dict[str, float]
):
    cfg = Config.get_config(
        {
            "model": {"predefined_model": {"name": model_name}},
            "trainer": {
                "batch_size": 2,
                "callbacks": [
                    {
                        "name": "GradientAccumulationScheduler",
                        "params": {"scheduling": {}},
                    }
                ],
            },
        }
    )

    assert cfg.model.predefined_model is not None
    assert cfg.model.predefined_model.params["loss_params"] == expected_weights
    assert cfg.trainer.callbacks[0].params["scheduling"] == {
        0: 1,
        1: 16,
        2: 32,
    }


def test_smart_cfg_auto_populate_rejects_invalid_loss_params():
    predefined_model = PredefinedModelConfig.model_construct(
        name="DetectionModel", params={"loss_params": ["bad"]}
    )
    cfg = Config.model_construct(
        model=ModelConfig.model_construct(predefined_model=predefined_model),
        trainer=TrainerConfig(batch_size=2),
        loader=LoaderConfig(),
    )

    with pytest.raises(ValueError, match="loss_params"):
        cfg.smart_auto_populate()


def test_predefined_model_loading_can_exclude_attachments():
    cfg = Config.get_config(
        {
            "model": {
                "predefined_model": {
                    "name": "DetectionModel",
                    "include_losses": False,
                    "include_metrics": False,
                    "include_visualizers": False,
                }
            },
            "trainer": {"smart_cfg_auto_populate": False},
        }
    )

    head = cfg.model.nodes[-1]
    assert head.losses == []
    assert head.metrics == []
    assert head.visualizers == []


def test_list_predefined_models_covers_known_presets():
    entries = list_predefined_models()

    assert "detection" in entries
    assert "anomaly_detection" in entries
    assert "embeddings" in entries

    assert entries["detection"][0] in VARIANT_ORDER
    assert set(entries["detection"]) == {"light", "heavy"}

    assert entries["anomaly_detection"] == [None]
    assert entries["embeddings"] == [None]

    for excluded in (
        "defaults",
        "complex",
        "example_export",
        "example_tuning",
    ):
        assert excluded not in entries


def test_resolve_predefined_config_variants():
    resolved = resolve_predefined_config("detection", "light")
    assert isinstance(resolved.path, Path)
    assert resolved.path.exists()
    assert resolved.path.name == "detection_light_model.yaml"
    assert resolved.opts == []

    default = resolve_predefined_config("detection", None)
    expected_variant = list_predefined_models()["detection"][0]
    assert default.path.name == f"detection_{expected_variant}_model.yaml"

    variantless = resolve_predefined_config("anomaly_detection", None)
    assert variantless.path.name == "anomaly_detection_model.yaml"

    versioned = resolve_predefined_config("detection:v1", "light")
    assert versioned.opts == ["model.predefined_model.version", "1"]

    latest = resolve_predefined_config("detection:latest", "light")
    assert latest.opts == []

    with pytest.raises(ValueError, match="Malformed model spec"):
        resolve_predefined_config("detection:bad", None)

    with pytest.raises(ValueError, match="Unknown predefined model 'nope'"):
        resolve_predefined_config("nope", None)

    with pytest.raises(
        ValueError,
        match="Variant 'nope' is not available for model 'detection'",
    ):
        resolve_predefined_config("detection", "nope")


def test_predefined_metaclass_keys_and_aliases_shipped_models():
    assert MODELS._module_dict["DetectionModel:v1"] is DetectionModel
    assert MODELS._module_dict["DetectionModel"] is DetectionModel
    assert "SimplePredefinedModel" not in MODELS._module_dict
    assert "BasePredefinedModel" not in MODELS._module_dict


def test_predefined_metaclass_keys_versioned_class_to_family(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(MODELS, "_module_dict", dict(MODELS._module_dict))

    class DetectionModelV2(DetectionModel):
        _VERSION = 2

    assert "DetectionModelV2" not in MODELS._module_dict
    assert MODELS._module_dict["DetectionModel:v2"] is DetectionModelV2
    assert resolve_predefined_class("DetectionModel", 2) is DetectionModelV2
    assert (
        resolve_predefined_class("DetectionModel", "latest")
        is DetectionModelV2
    )


def test_predefined_version_resolver_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    registry: dict[str, Any] = {
        "DetectionModel:v1": DetectionModel,
        "DetectionModel": DetectionModel,
        "ConcreteSimplePredefinedModel": ConcreteSimplePredefinedModel,
        "NoVersionPreset": object(),
    }
    monkeypatch.setattr(MODELS, "_module_dict", registry)

    class DetectionModelV2(DetectionModel):
        _VERSION = 2

    assert _split_family_version("DetectionModel:v2") == ("DetectionModel", 2)
    assert _split_family_version("DetectionModel") == ("DetectionModel", None)
    assert list_versions("ConcreteSimplePredefinedModel") == {
        1: "ConcreteSimplePredefinedModel"
    }
    assert list_versions("NoVersionPreset") == {}

    assert resolve_predefined_class("DetectionModel") is DetectionModelV2
    assert resolve_predefined_class("DetectionModel", 1) is DetectionModel
    assert (
        resolve_predefined_class("DetectionModel:v2", "latest")
        is DetectionModelV2
    )
    assert resolved_class_name("DetectionModel") == "DetectionModel:v2"

    with pytest.raises(ValueError, match="conflicts with version=1"):
        resolve_predefined_class("DetectionModel:v2", 1)

    with pytest.raises(ValueError, match="No predefined model registered"):
        resolve_predefined_class("DoesNotExist")

    with pytest.raises(
        ValueError, match=r"Version 99.+Available versions: \[1, 2\]"
    ):
        resolve_predefined_class("DetectionModel", 99)

    current = PredefinedModelConfig(name="DetectionModel", version=2)
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(
            current, {"name": "DetectionModel", "version": 1}
        )
    assert warn.call_count == 1

    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(None, {"name": "DetectionModel"})
        warn_on_predefined_model_mismatch(current, None)
        warn_on_predefined_model_mismatch(current, {"version": 1})
    assert warn.call_count == 0


@pytest.mark.parametrize(
    ("model_cls", "expected_default", "expected_nodes"),
    [
        (AnomalyDetectionModel, "light", ["RecSubNet", "DiscSubNetHead"]),
        (ClassificationModel, "light", ["ResNet", "ClassificationHead"]),
        (EmbeddingsModel, "default", ["GhostFaceNet", "GhostFaceNetHead"]),
        (
            DetectionModel,
            "light",
            ["EfficientRep", "RepPANNeck", "EfficientBBoxHead"],
        ),
        (FOMOModel, "light", ["EfficientRep", "FOMOHead"]),
        (
            InstanceSegmentationModel,
            "light",
            ["EfficientRep", "RepPANNeck", "PrecisionSegmentBBoxHead"],
        ),
        (
            KeypointDetectionModel,
            "light",
            ["EfficientRep", "RepPANNeck", "EfficientKeypointBBoxHead"],
        ),
        (
            OCRRecognitionModel,
            "light",
            ["PPLCNetV3", "SVTRNeck", "OCRCTCHead"],
        ),
    ],
)
def test_predefined_model_defaults_and_variants(
    model_cls: Any,
    expected_default: str,
    expected_nodes: list[str],
):
    default, variants = model_cls.get_variants()
    model = model_cls(variant="default")

    assert default == expected_default
    assert expected_default in variants
    assert [node.name for node in model.nodes] == expected_nodes


def test_simple_predefined_model_branches():
    model = ConcreteSimplePredefinedModel(
        backbone="Backbone",
        neck="Neck",
        head="Head",
        loss="Loss",
        metrics=["MeanAveragePrecision", "Accuracy"],
        main_metric="Accuracy",
        visualizer="Visualizer",
        confusion_matrix_available=True,
        backbone_params={"freezing": {"unfreeze_after": 1}},
        neck_params=cast(Any, {"freezing": FreezingConfig(active=True)}),
        head_params={"freezing": {"lr_after_unfreeze": 0.1}},
        metrics_params={"shared": True},
        visualizer_params={"alpha": 1},
        confusion_matrix_params={"normalize": True},
        task_name="task",
        torchmetrics_task="multiclass",
        per_class_metrics=True,
        finetuning={
            "backbone": [
                {"parameters": "conv", "optimizer": {"params": {"lr": 0.01}}}
            ],
            "head": [{"parameters": {"module_type": "Linear"}}],
        },
    )

    nodes = model.nodes
    assert [node.name for node in nodes] == ["Backbone", "Neck", "Head"]
    assert nodes[0].freezing.active is True
    assert nodes[1].freezing.active is True
    assert nodes[2].freezing.lr_after_unfreeze == 0.1
    assert nodes[2].task_name == "task"
    assert nodes[2].metrics[0].params["per_class_metrics"] is True
    assert nodes[2].metrics[1].params["per_class_metrics"] is True
    assert nodes[2].metrics[1].params["torchmetrics_task"] == "multiclass"
    assert nodes[2].metrics[1].is_main_metric is True
    assert nodes[2].metrics[2].name == "ConfusionMatrix"
    assert nodes[2].visualizers[0].params == {"alpha": 1}
    assert nodes[0].finetuning[0].parameters is not None
    assert nodes[0].finetuning[0].parameters[0].name == "conv"
    assert nodes[2].finetuning[0].parameters is not None
    assert nodes[2].finetuning[0].parameters[0].module_type == "Linear"

    without_neck = ConcreteSimplePredefinedModel(
        backbone="Backbone",
        neck="Neck",
        use_neck=False,
        head="Head",
        loss="Loss",
        metrics="Metric",
        visualizer=None,
        confusion_matrix_available=True,
        enable_confusion_matrix=False,
    )
    assert [node.name for node in without_neck.nodes] == ["Backbone", "Head"]
    assert without_neck.nodes[-1].inputs == ["Backbone"]
    assert without_neck.nodes[-1].visualizers == []
    assert (
        without_neck.generate_nodes(
            include_losses=False,
            include_metrics=False,
            include_visualizers=False,
        )[-1].losses
        == []
    )

    with pytest.raises(ValueError, match="exactly one metric"):
        ConcreteSimplePredefinedModel(
            backbone="Backbone",
            head="Head",
            loss="Loss",
            metrics=["A", "B"],
        )

    with pytest.raises(ValueError, match="freezing"):
        ConcreteSimplePredefinedModel._get_freezing({"freezing": "bad"})

    assert ConcreteSimplePredefinedModel._get_freezing({}).active is False


def test_ocr_recognition_model_alphabets_and_overrides():
    model = OCRRecognitionModel(
        alphabet="numeric",
        max_text_len=10,
        ignore_unknown=False,
    )
    head = model.nodes[-1]
    assert model.nodes[0].params["max_text_len"] == 10
    assert head.params["alphabet"] == list("0123456789")
    assert head.params["ignore_unknown"] is False

    custom = OCRRecognitionModel(
        alphabet=["x"],
        backbone_params={"max_text_len": 5},
        head_params={"alphabet": ["y"], "ignore_unknown": True},
    )
    assert custom.nodes[0].params["max_text_len"] == 5
    assert custom.nodes[-1].params["alphabet"] == ["y"]
    assert custom.nodes[-1].params["ignore_unknown"] is True

    assert OCRRecognitionModel._generate_alphabet("ascii")[0] == " "
    assert OCRRecognitionModel._generate_alphabet(["a", "b"]) == ["a", "b"]

    with pytest.raises(ValueError, match="Invalid alphabet"):
        OCRRecognitionModel._generate_alphabet(cast(Any, "bad"))


def test_segmentation_model_auxiliary_head_branches():
    model = SegmentationModel()
    assert [node.name for node in model.nodes] == [
        "DDRNet",
        "DDRNetSegmentationHead",
        "DDRNetSegmentationHead",
    ]
    assert model.nodes[-1].alias == "DDRNetSegmentationHead_aux"
    assert model.nodes[-1].remove_on_export is True
    assert model.nodes[-1].losses[0].weight == 0.4
    assert model.nodes[1].params["attach_index"] == -1
    assert model.nodes[2].params["attach_index"] == -2

    assert [
        node.name for node in SegmentationModel(use_aux_head=False).nodes
    ] == [
        "DDRNet",
        "DDRNetSegmentationHead",
    ]

    keep_aux = SegmentationModel(aux_head_params={"use_aux_heads": False})
    assert keep_aux.nodes[-1].remove_on_export is False

    with pytest.raises(TypeError, match="must be a boolean"):
        SegmentationModel(aux_head_params={"use_aux_heads": "bad"})


def test_pydantic_validation_errors_are_raised():
    with pytest.raises(ValidationError):
        ExportConfig.model_validate({"blobconverter": {"version": "bad"}})

    with pytest.raises(ValidationError):
        TrainerConfig(batch_size=0)

    with pytest.raises(ValidationError):
        AugmentationConfig.model_validate(
            {"name": "Invalid", "apply_on_stages": ["invalid"]}
        )


def test_simple_config_model_has_no_outputs_when_empty():
    model = ModelConfig()
    assert model.outputs == []


def test_versioned_name_still_triggers_smart_auto_populate():
    """Version pins must not bypass family-specific training
    defaults.
    """

    def build(name: str) -> Config:
        return Config.get_config(
            {
                "model": {
                    "predefined_model": {"name": name, "variant": "light"}
                },
                "trainer": {"batch_size": 4},
            }
        )

    plain = build("InstanceSegmentationModel")
    pinned = build("InstanceSegmentationModel:v1")
    assert plain.model.predefined_model is not None
    assert pinned.model.predefined_model is not None

    loss_params = cast(
        dict[str, Any], pinned.model.predefined_model.params["loss_params"]
    )
    assert loss_params == plain.model.predefined_model.params["loss_params"]
    assert loss_params["bbox_loss_weight"] == 7.5 * 16


def test_explicit_accumulate_grad_batches_is_not_overwritten():
    """Auto-population must preserve an explicit accumulation value."""
    cfg = Config.get_config(
        {
            "model": {
                "predefined_model": {
                    "name": "DetectionModel",
                    "variant": "light",
                }
            },
            "trainer": {"batch_size": 16, "accumulate_grad_batches": 1},
        }
    )
    assert cfg.trainer.accumulate_grad_batches == 1


def test_accumulate_grad_batches_auto_populated_when_unset():
    cfg = Config.get_config(
        {
            "model": {
                "predefined_model": {
                    "name": "DetectionModel",
                    "variant": "light",
                }
            },
            "trainer": {"batch_size": 16},
        }
    )
    assert cfg.trainer.accumulate_grad_batches == 4


def test_embeddings_preset_keeps_its_training_recipe():
    """The embeddings preset trains without gradient accumulation."""
    from luxonis_train.config.predefined import resolve_predefined_config

    cfg = Config.get_config(
        str(resolve_predefined_config("embeddings", None).path)
    )
    assert cfg.trainer.accumulate_grad_batches == 1


def test_embeddings_model_metadata_task_is_configurable():
    """The embeddings metadata field is dataset-specific."""
    from luxonis_train.config.predefined_models import EmbeddingsModel

    custom = cast(Any, EmbeddingsModel)(metadata_task_override="person_id")
    head = custom.nodes[1]
    assert head.metadata_task_override == "person_id"
    assert head.alias == "person_id-embeddings"

    default = cast(Any, EmbeddingsModel)()
    assert default.nodes[1].metadata_task_override == "color"
    assert default.nodes[1].alias == "color-embeddings"


def test_ocr_preset_keeps_its_explicit_accumulation():
    """The OCR preset deliberately pairs a small batch with explicit
    accumulation and gradient clipping.

    Auto-population used to overwrite the configured 2 with 64/4 = 16,
    quadrupling the effective batch the config asks for.
    """
    from luxonis_train.config.predefined import resolve_predefined_config

    cfg = Config.get_config(
        str(resolve_predefined_config("ocr_recognition", None).path)
    )
    assert cfg.trainer.batch_size == 4
    assert cfg.trainer.accumulate_grad_batches == 2


def test_model_class_returns_none_for_unknown_model():
    assert _model_class("nope") is None


def test_model_class_returns_none_for_config_without_predefined_model(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        predefined,
        "default_config_path",
        lambda _: configs_dir() / "defaults.yaml",
    )
    assert _model_class("detection") is None
    assert list_variants("detection") == ["light", "heavy"]


def test_model_class_returns_none_when_class_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
):
    from luxonis_train.config import predefined_versions

    def unresolvable(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("gone")

    monkeypatch.setattr(
        predefined_versions, "resolve_predefined_class", unresolvable
    )
    assert _model_class("detection") is None


def test_list_variants_skips_class_default_for_unvarianted_config():
    """A single unvarianted YAML already stands for the class
    default.
    """
    assert list_variants("embeddings") == [None]


def test_list_variants_includes_variants_without_a_yaml():
    assert list_variants("detection") == ["light", "medium", "heavy"]


def test_resolve_variant_without_dedicated_yaml():
    resolved = resolve_predefined_config("detection", "medium")
    assert resolved.path.name == "detection_light_model.yaml"
    assert resolved.opts == ["model.predefined_model.variant", "medium"]


def test_warn_silent_when_current_config_does_not_resolve():
    current = PredefinedModelConfig(name="DoesNotExist")
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(
            current, {"name": "DetectionModel", "version": 1}
        )
    assert warn.call_count == 0


def test_warn_when_checkpoint_model_no_longer_resolves():
    current = PredefinedModelConfig(name="DetectionModel", version=1)
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(
            current, {"name": "RemovedModel", "version": 1}
        )
    assert warn.call_count == 1
    assert "RemovedModel" in warn.call_args.args[0]
