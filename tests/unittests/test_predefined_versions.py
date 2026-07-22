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
    _split_family_version,
    list_versions,
    resolve_predefined_class,
    resolved_class_name,
    warn_on_predefined_model_mismatch,
)
from luxonis_train.lightning.luxonis_lightning import (
    LuxonisLightningModule,
    _checkpoint_predefined_model,
)
from luxonis_train.registry import MODELS


@pytest.fixture
def fake_v2_model() -> Iterator[type[DetectionModel]]:
    """Register a throw-away DetectionModel v2 for the duration of a
    test.
    """

    class DetectionModelV2(DetectionModel):
        _VERSION = 2

    # AutoRegisterMeta first registers the plain class name. Rekey it through
    # the production path to verify the documented ``FamilyV2`` convention.
    predefined_models._rekey_registry_with_versions()
    try:
        yield DetectionModelV2
    finally:
        MODELS._module_dict.pop("DetectionModel:v2", None)


@pytest.fixture
def plain_key_custom_model() -> Iterator[type[DetectionModel]]:
    class PlainKeyCustomModel(DetectionModel):
        _VERSION = 7

    MODELS._module_dict.pop("PlainKeyCustomModel:v7", None)
    try:
        yield PlainKeyCustomModel
    finally:
        MODELS._module_dict.pop("PlainKeyCustomModel", None)
        MODELS._module_dict.pop("PlainKeyCustomModel:v7", None)


def test_split_family_version():
    assert _split_family_version("DetectionModel:v1") == ("DetectionModel", 1)
    assert _split_family_version("DetectionModel:v17") == (
        "DetectionModel",
        17,
    )
    # Bare family name (no `:v` suffix) yields None version.
    assert _split_family_version("DetectionModel") == ("DetectionModel", None)


def test_resolve_latest_when_only_v1_exists():
    assert (
        resolve_predefined_class("DetectionModel", "latest") is DetectionModel
    )
    assert resolve_predefined_class("DetectionModel", 1) is DetectionModel


def test_resolve_defaults_to_latest():
    assert resolve_predefined_class("DetectionModel") is DetectionModel


def test_resolve_latest_picks_highest_version(
    fake_v2_model: type[DetectionModel],
):
    assert (
        resolve_predefined_class("DetectionModel", "latest") is fake_v2_model
    )
    assert resolve_predefined_class("DetectionModel", 2) is fake_v2_model
    assert resolve_predefined_class("DetectionModel", 1) is DetectionModel


def test_resolve_explicit_registry_key(fake_v2_model: type[DetectionModel]):
    assert (
        resolve_predefined_class("DetectionModel:v2", "latest")
        is fake_v2_model
    )
    assert resolve_predefined_class("DetectionModel:v2", 2) is fake_v2_model


def test_resolve_explicit_key_conflicting_version_errors(
    fake_v2_model: type[DetectionModel],
):
    with pytest.raises(ValueError, match="conflicts with version=1"):
        resolve_predefined_class("DetectionModel:v2", 1)


def test_resolve_unknown_family():
    with pytest.raises(ValueError, match="No predefined model registered"):
        resolve_predefined_class("DoesNotExist", "latest")


def test_resolve_unknown_version(fake_v2_model: type[DetectionModel]):
    with pytest.raises(
        ValueError, match=r"Version 99.+Available versions: \[1, 2\]"
    ):
        resolve_predefined_class("DetectionModel", 99)


def test_list_versions_returns_registered_keys(
    fake_v2_model: type[DetectionModel],
):
    assert list_versions("DetectionModel") == {
        1: "DetectionModel:v1",
        2: "DetectionModel:v2",
    }


def test_resolve_plain_key_custom_predefined_model(
    plain_key_custom_model: type[DetectionModel],
):
    assert list_versions("PlainKeyCustomModel") == {7: "PlainKeyCustomModel"}
    assert (
        resolve_predefined_class("PlainKeyCustomModel")
        is plain_key_custom_model
    )
    assert (
        resolve_predefined_class("PlainKeyCustomModel", 7)
        is plain_key_custom_model
    )
    assert resolved_class_name("PlainKeyCustomModel") == "PlainKeyCustomModel"


def test_resolved_class_name_uses_colon_format(
    fake_v2_model: type[DetectionModel],
):
    assert (
        resolved_class_name("DetectionModel", "latest") == "DetectionModel:v2"
    )
    assert resolved_class_name("DetectionModel", 1) == "DetectionModel:v1"


def test_predefined_model_config_defaults_version_to_latest():
    cfg = PredefinedModelConfig(name="DetectionModel")
    assert cfg.version == "latest"


def test_predefined_model_config_rejects_invalid_version():
    with pytest.raises(ValueError, match="version"):
        PredefinedModelConfig(
            name="DetectionModel", version=cast(Any, "not-a-version")
        )


def test_predefined_model_config_accepts_int_and_latest():
    assert PredefinedModelConfig(name="DetectionModel", version=1).version == 1
    assert (
        PredefinedModelConfig(name="DetectionModel", version="latest").version
        == "latest"
    )


def test_warn_fires_on_mismatch(fake_v2_model: type[DetectionModel]):
    current = PredefinedModelConfig(name="DetectionModel", version=2)
    ckpt_pm = {"name": "DetectionModel", "version": 1}
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(current, ckpt_pm)
    assert warn.call_count == 1
    msg = warn.call_args.args[0]
    assert "DetectionModel:v2" in msg
    assert "DetectionModel:v1" in msg


def test_warn_silent_on_match(fake_v2_model: type[DetectionModel]):
    current = PredefinedModelConfig(name="DetectionModel", version=2)
    ckpt_pm = {"name": "DetectionModel", "version": 2}
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(current, ckpt_pm)
    assert warn.call_count == 0


def test_warn_silent_when_no_predefined_model_in_ckpt():
    current = PredefinedModelConfig(name="DetectionModel", version=1)
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(current, None)
        warn_on_predefined_model_mismatch(current, "not-a-dict")
    assert warn.call_count == 0


def test_warn_silent_when_no_current_predefined_model():
    ckpt_pm = {"name": "DetectionModel", "version": 1}
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(None, ckpt_pm)
    assert warn.call_count == 0


def test_checkpoint_metadata_includes_excluded_predefined_model():
    cfg = Config.get_config(
        {
            "model": {
                "predefined_model": {
                    "name": "DetectionModel",
                    "version": 1,
                }
            },
            "trainer": {"smart_cfg_auto_populate": False},
        }
    )

    assert "predefined_model" not in cfg.model_dump()["model"]
    ckpt_predefined_model = _checkpoint_predefined_model(cfg)
    assert ckpt_predefined_model is not None
    assert ckpt_predefined_model["name"] == "DetectionModel"
    assert ckpt_predefined_model["version"] == 1


def test_lightning_warn_uses_checkpoint_predefined_model_metadata(
    fake_v2_model: type[DetectionModel],
):
    module = cast(Any, LuxonisLightningModule.__new__(LuxonisLightningModule))
    module.cfg = SimpleNamespace(
        model=SimpleNamespace(
            predefined_model=PredefinedModelConfig(
                name="DetectionModel", version=2
            )
        )
    )

    with patch.object(logger, "warning") as warn:
        module._warn_on_predefined_model_mismatch(
            {"model": {}}, {"name": "DetectionModel", "version": 1}
        )
    assert warn.call_count == 1


def test_all_shipped_predefined_models_use_colon_key():
    """Every registered predefined model uses the `Family:vN` key
    format.
    """
    for name in predefined_models.__all__:
        if name == "BasePredefinedModel":
            continue
        cls = getattr(predefined_models, name)
        key = f"{cls.__name__}:v{cls._VERSION}"
        assert MODELS._module_dict.get(key) is cls
        assert cls.__name__ not in MODELS._module_dict

        family, _, version_part = key.partition(":")
        assert family, f"empty family in registry key: {key!r}"
        assert version_part.startswith("v"), (
            f"registry key {key!r} does not use `:vN` format"
        )
        assert version_part[1:].isdigit(), (
            f"registry key {key!r} does not use `:vN` format"
        )
