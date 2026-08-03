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
from luxonis_train.lightning import luxonis_lightning
from luxonis_train.lightning.luxonis_lightning import (
    LuxonisLightningModule,
    _checkpoint_predefined_model,
)
from luxonis_train.registry import MODELS


@pytest.fixture
def fake_v2_model() -> Iterator[type[DetectionModel]]:
    """Register a throw-away DetectionModel v2 for the duration of a
    test.

    Defining the class is all it takes: the metaclass keys it as
    ``DetectionModel:v2`` right away, which is what makes the documented
    ``FamilyV2`` convention work for models loaded after startup.
    """
    previous_alias = MODELS._module_dict.get("DetectionModel")

    class DetectionModelV2(DetectionModel):
        _VERSION = 2

    try:
        yield DetectionModelV2
    finally:
        MODELS._module_dict.pop("DetectionModel:v2", None)
        if previous_alias is not None:
            MODELS._module_dict["DetectionModel"] = previous_alias


@pytest.fixture
def plain_key_custom_model() -> Iterator[type[DetectionModel]]:
    """Register a model reachable only under a plain registry key."""

    class PlainKeyCustomModel(DetectionModel):
        _VERSION = 7

    # Drop the versioned key so only the plain alias is left, standing
    # in for a model registered outside the versioned key scheme.
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


def test_checkpoint_metadata_pins_latest_to_concrete_version(
    fake_v2_model: type[DetectionModel],
):
    """`version: "latest"` must be resolved to a concrete integer at
    checkpoint time.

    Otherwise a checkpoint trained today against v1 becomes
    indistinguishable from a future v2 checkpoint (both stored as
    ``"latest"``), and the mismatch warning is silently suppressed once
    the default flips.
    """
    cfg = Config.get_config(
        {
            "model": {
                "predefined_model": {
                    "name": "DetectionModel",
                    "version": "latest",
                }
            },
            "trainer": {"smart_cfg_auto_populate": False},
        }
    )
    ckpt_predefined_model = _checkpoint_predefined_model(cfg)
    assert ckpt_predefined_model is not None
    assert ckpt_predefined_model["version"] == 2


def _module_with_predefined_model(version: int) -> Any:
    module = cast(Any, LuxonisLightningModule.__new__(LuxonisLightningModule))
    module.cfg = SimpleNamespace(
        model=SimpleNamespace(
            predefined_model=PredefinedModelConfig(
                name="DetectionModel", version=version
            )
        )
    )
    return module


def test_lightning_warn_uses_top_level_metadata(
    fake_v2_model: type[DetectionModel],
):
    module = _module_with_predefined_model(2)
    with patch.object(logger, "warning") as warn:
        module._warn_on_predefined_model_mismatch(
            {"name": "DetectionModel", "version": 1}
        )
    assert warn.call_count == 1


def test_lightning_warn_silent_for_checkpoint_without_pin(
    fake_v2_model: type[DetectionModel],
):
    """Checkpoints written before versioning carry no pin at all."""
    module = _module_with_predefined_model(2)
    with patch.object(logger, "warning") as warn:
        module._warn_on_predefined_model_mismatch(None)
    assert warn.call_count == 0


def test_warn_when_checkpoint_family_no_longer_resolves():
    """A checkpoint naming a family that is gone must warn, not go
    quiet.

    This is the "breaking architecture change" case the warning exists
    for; swallowing the resolution error leaves the user with an opaque
    state-dict failure instead.
    """
    current = PredefinedModelConfig(name="DetectionModel", version=1)
    ckpt_pm = {"name": "RemovedModel", "version": 1}
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(current, ckpt_pm)
    assert warn.call_count == 1
    assert "RemovedModel" in warn.call_args.args[0]


def test_warn_silent_when_checkpoint_entry_has_no_name():
    current = PredefinedModelConfig(name="DetectionModel", version=1)
    with patch.object(logger, "warning") as warn:
        warn_on_predefined_model_mismatch(current, {"version": 1})
    assert warn.call_count == 0


def test_checkpoint_pin_survives_resaving_without_config():
    """Re-saving a run rebuilt from a checkpoint must keep the pin.

    `predefined_model` is excluded from the dumped config, so a
    `--weights`-only run has none in `cfg`; without carrying the loaded
    value over, `upgrade checkpoint` would silently strip it.
    """
    module = cast(Any, LuxonisLightningModule.__new__(LuxonisLightningModule))
    module.cfg = SimpleNamespace(
        model=SimpleNamespace(predefined_model=None),
        model_dump=dict,
    )
    module.dataset_metadata = SimpleNamespace(dump=dict)
    pin = {"name": "DetectionModel", "version": 1}
    module._ckpt_predefined_model = pin

    checkpoint: dict[str, Any] = {"state_dict": {}}
    with (
        patch.object(
            luxonis_lightning, "filter_checkpoint_state_dict", lambda sd: sd
        ),
        patch.object(
            luxonis_lightning, "get_model_execution_order", lambda _: []
        ),
    ):
        module._add_custom_data_to_checkpoint(checkpoint)

    assert checkpoint["predefined_model"] == pin


def test_checkpoint_has_no_pin_when_none_is_known():
    module = cast(Any, LuxonisLightningModule.__new__(LuxonisLightningModule))
    module.cfg = SimpleNamespace(
        model=SimpleNamespace(predefined_model=None),
        model_dump=dict,
    )
    module.dataset_metadata = SimpleNamespace(dump=dict)

    checkpoint: dict[str, Any] = {"state_dict": {}}
    with (
        patch.object(
            luxonis_lightning, "filter_checkpoint_state_dict", lambda sd: sd
        ),
        patch.object(
            luxonis_lightning, "get_model_execution_order", lambda _: []
        ),
    ):
        module._add_custom_data_to_checkpoint(checkpoint)

    assert "predefined_model" not in checkpoint


def test_all_shipped_predefined_models_are_addressable():
    """Every predefined model is registered under `Family:vN` and under
    its plain class name.
    """
    for name in predefined_models.__all__:
        if name == "BasePredefinedModel":
            continue
        cls = getattr(predefined_models, name)
        key = f"{cls.__name__}:v{cls._VERSION}"
        assert MODELS._module_dict.get(key) is cls
        # The plain alias is kept so that looking a predefined model up
        # by its class name keeps working.
        assert MODELS._module_dict.get(cls.__name__) is cls

        family, _, version_part = key.partition(":")
        assert family, f"empty family in registry key: {key!r}"
        assert version_part.startswith("v"), (
            f"registry key {key!r} does not use `:vN` format"
        )
        assert version_part[1:].isdigit(), (
            f"registry key {key!r} does not use `:vN` format"
        )


def test_abstract_intermediates_are_not_registered():
    assert "SimplePredefinedModel" not in MODELS._module_dict
    assert "BasePredefinedModel" not in MODELS._module_dict


def test_custom_model_overrides_shipped_family():
    """Registering a class under a built-in's name must take effect.

    Before versioning, `AutoRegisterMeta`'s force-overwrite meant a
    custom model loaded through `--source` replaced the shipped one.
    Keying the shipped class as `Family:vN` must not turn that into a
    silent no-op.
    """
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
    """A model registered under a plain key still gets a concrete
    version.

    The version has to come off the resolved class, not off the registry
    key - a plain key has no `:vN` suffix to parse.
    """
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
    ckpt_predefined_model = _checkpoint_predefined_model(cfg)
    assert ckpt_predefined_model is not None
    assert ckpt_predefined_model["version"] == 7
