from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from loguru import logger

from luxonis_train.config import predefined_models
from luxonis_train.config.config import Config, PredefinedModelConfig
from luxonis_train.config.predefined_models import DetectionModel
from luxonis_train.config.predefined_versions import (
    resolve_predefined_class,
    warn_on_predefined_model_mismatch,
)
from luxonis_train.core.utils.export_utils import replace_weights
from luxonis_train.lightning import luxonis_lightning
from luxonis_train.lightning.luxonis_lightning import (
    LuxonisLightningModule,
    _checkpoint_predefined_model,
)
from luxonis_train.registry import MODELS


@pytest.fixture
def fake_v2_model() -> Iterator[type[DetectionModel]]:
    previous_alias = MODELS._module_dict.get("DetectionModel")
    previous_latest = MODELS._module_dict.get("DetectionModel:latest")

    class DetectionModelV2(DetectionModel, register_name="DetectionModel"):
        _VERSION = 2

    try:
        yield DetectionModelV2
    finally:
        MODELS._module_dict.pop("DetectionModel:v2", None)
        if previous_alias is not None:
            MODELS._module_dict["DetectionModel"] = previous_alias
        if previous_latest is not None:
            MODELS._module_dict["DetectionModel:latest"] = previous_latest


@pytest.fixture
def plain_key_custom_model() -> Iterator[type[DetectionModel]]:
    class PlainKeyCustomModel(DetectionModel):
        _VERSION = 7

    MODELS._module_dict.pop("PlainKeyCustomModel:v7", None)
    MODELS._module_dict.pop("PlainKeyCustomModel:latest", None)
    try:
        yield PlainKeyCustomModel
    finally:
        MODELS._module_dict.pop("PlainKeyCustomModel", None)
        MODELS._module_dict.pop("PlainKeyCustomModel:v7", None)


def test_predefined_model_config_version_validation():
    assert PredefinedModelConfig(name="DetectionModel").version == "latest"
    assert PredefinedModelConfig(name="DetectionModel", version=1).version == 1
    with pytest.raises(ValueError, match="version"):
        PredefinedModelConfig(
            name="DetectionModel", version=cast(Any, "not-a-version")
        )


def test_matching_checkpoint_version_does_not_warn(
    fake_v2_model: type[DetectionModel],
):
    current = PredefinedModelConfig(name="DetectionModel", version=2)
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(
            current, {"name": "DetectionModel", "version": 2}
        )
    warn.assert_not_called()


@pytest.mark.parametrize(("version", "expected"), [(1, 1), ("latest", 2)])
def test_checkpoint_metadata_pins_predefined_model_version(
    version: int | str,
    expected: int,
    fake_v2_model: type[DetectionModel],
):
    cfg = Config.get_config(
        {
            "model": {
                "predefined_model": {
                    "name": "DetectionModel",
                    "version": version,
                }
            },
            "trainer": {"smart_cfg_auto_populate": False},
        }
    )

    assert "predefined_model" not in cfg.model_dump()["model"]
    metadata = _checkpoint_predefined_model(cfg)
    assert metadata is not None
    assert metadata["name"] == "DetectionModel"
    assert metadata["version"] == expected


def _checkpoint_module(
    pin: dict[str, Any] | None = None,
) -> LuxonisLightningModule:
    module = cast(Any, LuxonisLightningModule.__new__(LuxonisLightningModule))
    module.cfg = SimpleNamespace(
        model=SimpleNamespace(predefined_model=None),
        model_dump=dict,
    )
    module.dataset_metadata = SimpleNamespace(dump=dict)
    module._ckpt_predefined_model = pin
    return module


def _add_checkpoint_metadata(
    module: LuxonisLightningModule, checkpoint: dict[str, Any]
) -> None:
    with (
        patch.object(
            luxonis_lightning, "filter_checkpoint_state_dict", lambda sd: sd
        ),
        patch.object(
            luxonis_lightning, "get_model_execution_order", lambda _: []
        ),
    ):
        module._add_custom_data_to_checkpoint(checkpoint)


def test_checkpoint_pin_survives_resaving_without_config():
    pin = {"name": "DetectionModel", "version": 1}
    checkpoint: dict[str, Any] = {"state_dict": {}}

    _add_checkpoint_metadata(_checkpoint_module(pin), checkpoint)

    assert checkpoint["predefined_model"] == pin


def test_temporary_weight_swap_restores_checkpoint_pin_before_resave():
    """A configless module restored to checkpoint A's weights must not
    save checkpoint B's pin after a temporary weight swap.
    """
    original_pin = {"name": "DetectionModel", "version": 1}
    temporary_pin = {"name": "DetectionModel", "version": 2}
    module = _checkpoint_module(original_pin)
    original_state = {"nodes.backbone.weight": object()}

    def load_checkpoint(checkpoint: dict[str, Any]) -> None:
        module._ckpt_predefined_model = checkpoint.get("predefined_model")

    with (
        patch.object(
            LuxonisLightningModule,
            "state_dict",
            return_value=original_state,
        ),
        patch.object(
            LuxonisLightningModule,
            "load_checkpoint",
            side_effect=load_checkpoint,
        ),
        patch.object(
            LuxonisLightningModule, "load_state_dict"
        ) as restore_state,
    ):
        with replace_weights(
            module,
            {"state_dict": {}, "predefined_model": temporary_pin},
        ):
            assert module._ckpt_predefined_model == temporary_pin

        restore_state.assert_called_once_with(original_state)

    checkpoint: dict[str, Any] = {"state_dict": {}}
    _add_checkpoint_metadata(module, checkpoint)

    assert checkpoint["predefined_model"] == original_pin


def test_checkpoint_removes_stale_pin_when_none_is_known():
    checkpoint: dict[str, Any] = {
        "state_dict": {},
        "predefined_model": {"name": "stale"},
    }

    _add_checkpoint_metadata(_checkpoint_module(), checkpoint)

    assert "predefined_model" not in checkpoint


def test_all_shipped_predefined_models_are_addressable():
    for name in predefined_models.__all__:
        if name == "BasePredefinedModel":
            continue
        model_cls = getattr(predefined_models, name)
        assert (
            MODELS._module_dict[f"{name}:v{model_cls._VERSION}"] is model_cls
        )
        assert MODELS._module_dict[name] is model_cls


def test_custom_model_overrides_shipped_family():
    shipped = MODELS._module_dict["DetectionModel:v1"]
    try:

        class DetectionModel(predefined_models.BasePredefinedModel):
            @staticmethod
            def get_variants() -> tuple[str, dict[str, Any]]:
                return "default", {"default": {}}

            @property
            def nodes(self) -> list[Any]:
                return []

        assert resolve_predefined_class("DetectionModel") is DetectionModel
        assert MODELS._module_dict["DetectionModel:v1"] is DetectionModel
        assert MODELS._module_dict["DetectionModel"] is DetectionModel
    finally:
        MODELS._module_dict["DetectionModel:v1"] = shipped
        MODELS._module_dict["DetectionModel"] = shipped


def test_checkpoint_metadata_pins_plain_key_model_version(
    plain_key_custom_model: type[DetectionModel],
):
    cfg = Config.get_config(
        {
            "model": {
                "predefined_model": {
                    "name": "PlainKeyCustomModel",
                    "version": "latest",
                }
            },
            "trainer": {"smart_cfg_auto_populate": False},
        }
    )

    metadata = _checkpoint_predefined_model(cfg)
    assert metadata is not None
    assert metadata["version"] == 7
