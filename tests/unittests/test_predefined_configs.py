import subprocess
import sys
from pathlib import Path

import pytest

from luxonis_train.__main__ import _resolve_config, _split_model_version
from luxonis_train.config.predefined import (
    VARIANT_ORDER,
    list_predefined_models,
    resolve_predefined_config,
)


def test_list_predefined_models_covers_known_presets():
    entries = list_predefined_models()

    # A few filenames we know ship with the package.
    assert "detection" in entries
    assert "anomaly_detection" in entries
    assert "embeddings" in entries

    assert entries["detection"][0] in VARIANT_ORDER
    assert set(entries["detection"]) == {"light", "heavy"}

    # Single-config models expose exactly one `None` "default" entry.
    assert entries["anomaly_detection"] == [None]
    assert entries["embeddings"] == [None]

    # No example/defaults files leak through.
    for excluded in (
        "defaults",
        "complex",
        "example_export",
        "example_tuning",
    ):
        assert excluded not in entries


def test_resolve_predefined_config_returns_existing_file():
    path = resolve_predefined_config("detection", "light")
    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "detection_light_model.yaml"


def test_resolve_predefined_config_defaults_variant():
    default = resolve_predefined_config("detection", None)
    # The default variant is the first one in list_predefined_models.
    entries = list_predefined_models()
    expected_variant = entries["detection"][0]
    assert default.name == f"detection_{expected_variant}_model.yaml"


def test_resolve_predefined_config_handles_variantless_model():
    path = resolve_predefined_config("anomaly_detection", None)
    assert path.name == "anomaly_detection_model.yaml"


def test_resolve_predefined_config_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown predefined model 'nope'"):
        resolve_predefined_config("nope", None)


def test_resolve_predefined_config_rejects_unknown_variant():
    with pytest.raises(
        ValueError,
        match="Variant 'nope' is not available for model 'detection'",
    ):
        resolve_predefined_config("detection", "nope")


def test_cli_resolver_passthrough_for_plain_config():
    assert _resolve_config("foo.yaml", None, None) == "foo.yaml"
    assert _resolve_config(None, None, None) is None


def test_cli_resolver_variant_without_model_errors():
    with pytest.raises(ValueError, match="'--variant' requires '--model'"):
        _resolve_config(None, None, "light")


def test_cli_resolver_config_and_model_mutually_exclusive():
    with pytest.raises(
        ValueError, match="'--config' and '--model' are mutually exclusive"
    ):
        _resolve_config("foo.yaml", "detection", None)


def test_cli_resolver_returns_packaged_path_for_model():
    resolved = _resolve_config(None, "detection", "light")
    assert isinstance(resolved, str)
    assert resolved.endswith("detection_light_model.yaml")
    assert Path(resolved).exists()


def test_split_model_version_forms():
    assert _split_model_version("detection") == ("detection", None)
    assert _split_model_version("detection:v1") == ("detection", "1")
    assert _split_model_version("detection:v12") == ("detection", "12")
    assert _split_model_version("detection:latest") == ("detection", "latest")


def test_split_model_version_rejects_malformed():
    with pytest.raises(ValueError, match="Malformed model spec"):
        _split_model_version("detection:bad")
    with pytest.raises(ValueError, match="Malformed model spec"):
        _split_model_version("detection:2")  # missing the leading `v`


def test_resolver_strips_version_before_yaml_lookup():
    """CLI `--model detection:v1` should still resolve to the same YAML
    preset as `--model detection`; the version part is consumed by
    `create_model`, not by the YAML resolver.
    """
    plain = _resolve_config(None, "detection", "light")
    with_version = _resolve_config(None, "detection:v1", "light")
    assert plain == with_version


def test_list_models_cli_command_runs_and_lists_models():
    """Smoke test the `list-models` command exposed on the CLI.

    Runs it in a subprocess so we exercise the real cyclopts wiring.
    """
    result = subprocess.run(
        [sys.executable, "-m", "luxonis_train", "list-models"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "detection" in result.stdout
    assert "anomaly_detection" in result.stdout
    # Default marker present.
    assert "*" in result.stdout
