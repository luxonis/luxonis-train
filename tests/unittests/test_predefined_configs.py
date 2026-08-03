import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from luxonis_train.__main__ import (
    _resolve_config,
    _split_model_version,
    create_model,
)
from luxonis_train.config.predefined import (
    CONFIGS_PACKAGE,
    VARIANT_ORDER,
    list_predefined_models,
    list_variants,
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
    resolved = resolve_predefined_config("detection", "light")
    assert isinstance(resolved.path, Path)
    assert resolved.path.exists()
    assert resolved.path.name == "detection_light_model.yaml"
    assert resolved.opts == []


def test_resolve_predefined_config_defaults_variant():
    default = resolve_predefined_config("detection", None)
    # The default variant is the first one in list_predefined_models.
    entries = list_predefined_models()
    expected_variant = entries["detection"][0]
    assert default.path.name == f"detection_{expected_variant}_model.yaml"


def test_resolve_predefined_config_handles_variantless_model():
    resolved = resolve_predefined_config("anomaly_detection", None)
    assert resolved.path.name == "anomaly_detection_model.yaml"


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
    assert _resolve_config("foo.yaml", None, None) == ("foo.yaml", [])
    assert _resolve_config(None, None, None) == (None, [])


def test_cli_resolver_variant_without_model_errors():
    with pytest.raises(ValueError, match="'--variant' requires '--model'"):
        _resolve_config(None, None, "light")


def test_cli_resolver_config_and_model_mutually_exclusive():
    with pytest.raises(
        ValueError, match="'--config' and '--model' are mutually exclusive"
    ):
        _resolve_config("foo.yaml", "detection", None)


def test_cli_resolver_returns_packaged_path_for_model():
    resolved, opts = _resolve_config(None, "detection", "light")
    assert isinstance(resolved, str)
    assert resolved.endswith("detection_light_model.yaml")
    assert Path(resolved).exists()
    assert opts == []


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
    assert "Variants" in result.stdout
    assert "Versions" in result.stdout
    # Default marker present.
    assert "*" in result.stdout


def test_info_cli_command_displays_model_components():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "luxonis_train",
            "info",
            "--model",
            "detection",
            "--variant",
            "light",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "DetectionModel:v1" in result.stdout
    assert "Backbone" in result.stdout
    assert "EfficientRep" in result.stdout
    assert "Neck" in result.stdout
    assert "RepPANNeck" in result.stdout
    assert "Head" in result.stdout
    assert "EfficientBBoxHead" in result.stdout


def _iter_predefined_model_variant_pairs() -> list[tuple[str, str | None]]:
    return [
        (model, variant)
        for model in list_predefined_models()
        for variant in list_variants(model)
    ]


@pytest.mark.parametrize(
    ("model", "variant"), _iter_predefined_model_variant_pairs()
)
def test_info_cli_runs_for_every_predefined_model_variant(
    model: str, variant: str | None
):
    args = [sys.executable, "-m", "luxonis_train", "info", "--model", model]
    if variant is not None:
        args += ["--variant", variant]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert result.stdout.strip(), (
        f"`info` produced no output for --model {model} --variant {variant!r}"
    )


def test_embeddings_model_uses_a_predefined_model_class():
    config = resolve_predefined_config("embeddings", None)
    assert "name: EmbeddingsModel" in config.path.read_text()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "luxonis_train",
            "info",
            "--model",
            "embeddings",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "EmbeddingsModel:v1" in result.stdout
    assert "GhostFaceNet" in result.stdout
    assert "GhostFaceNetHead" in result.stdout


def test_configs_are_packaged_inside_luxonis_train():
    """The presets must not ship as a generic top-level `configs`
    package.

    Any other `configs` directory on `sys.path` - a user's own project
    layout, most commonly - would win the import lookup and shadow them,
    and another distribution shipping that name would clobber the files
    outright.
    """
    assert CONFIGS_PACKAGE == "luxonis_train.configs"
    module = importlib.import_module(CONFIGS_PACKAGE)
    assert module.__name__ == "luxonis_train.configs"
    assert resolve_predefined_config("detection", "light").path.exists()


def test_class_only_variant_is_selectable():
    """`medium` ships no YAML of its own but is a real `DetectionModel`
    variant, so `--variant medium` has to reach it.
    """
    assert "medium" in list_variants("detection")
    assert "medium" not in list_predefined_models()["detection"]

    resolved = resolve_predefined_config("detection", "medium")
    assert resolved.path.name == "detection_light_model.yaml"
    assert resolved.opts == ["model.predefined_model.variant", "medium"]


def test_cli_resolver_passes_variant_override_through():
    resolved, opts = _resolve_config(None, "detection", "medium")
    assert isinstance(resolved, str)
    assert resolved.endswith("detection_light_model.yaml")
    assert opts == ["model.predefined_model.variant", "medium"]


def test_unknown_variant_error_lists_class_variants():
    with pytest.raises(ValueError, match="light, medium, heavy"):
        resolve_predefined_config("detection", "nope")


def test_split_model_version_rejects_non_ascii_digits():
    """`str.isdigit()` accepts superscripts and non-ASCII decimals that
    `int()` then rejects.
    """
    for spec in ("detection:v²", "detection:v٣"):
        with pytest.raises(ValueError, match="Malformed model spec"):
            _split_model_version(spec)


def test_create_model_requires_a_config_model_or_weights():
    """Commands must not fall through to an all-defaults model.

    `--config` is optional on every command, but with no `--model` and
    no `--weights` there is nothing to build from.
    """
    with pytest.raises(ValueError, match="No model source given"):
        create_model(None)
    # Bare `opts` used to satisfy `luxonis_ml`'s "cfg or overrides" check
    # and silently produce a default model.
    with pytest.raises(ValueError, match="No model source given"):
        create_model(None, ["trainer.epochs", "1"])
