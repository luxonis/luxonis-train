import importlib
import subprocess
import sys
from unittest.mock import patch

import pytest
from luxonis_ml.typing import Params

from luxonis_train import LuxonisModel
from luxonis_train.__main__ import create_model, info
from luxonis_train.config import Config
from luxonis_train.config.predefined import (
    CONFIGS_PACKAGE,
    list_predefined_models,
    list_variants,
    parse_model_spec,
    resolve_predefined_config,
)


def test_resolver_returns_packaged_config_and_overrides():
    resolved = resolve_predefined_config("detection:v1", "medium")

    assert resolved.path.name == "detection_light_model.yaml"
    assert resolved.path.exists()
    assert resolved.opts == [
        "model.predefined_model.version",
        "1",
        "model.predefined_model.variant",
        "medium",
    ]


@pytest.mark.parametrize(
    ("opts", "expected_opts"),
    [
        (
            ["trainer.epochs", "1"],
            [
                "trainer.epochs",
                "1",
                "model.predefined_model.version",
                "1",
                "model.predefined_model.variant",
                "medium",
            ],
        ),
        (
            {
                "trainer.epochs": 1,
                "model.predefined_model.variant": "heavy",
            },
            {
                "trainer.epochs": 1,
                "model.predefined_model.version": "1",
                "model.predefined_model.variant": "medium",
            },
        ),
    ],
)
def test_luxonis_model_resolves_packaged_config(
    opts: list[str] | Params,
    expected_opts: list[str] | Params,
):
    with (
        patch.object(
            Config, "get_config", side_effect=RuntimeError("config resolved")
        ) as get_config,
        pytest.raises(RuntimeError, match="config resolved"),
    ):
        LuxonisModel(
            model="detection:v1",
            variant="medium",
            opts=opts,
        )

    config_path, resolved_opts = get_config.call_args.args
    assert config_path.name == "detection_light_model.yaml"
    assert resolved_opts == expected_opts


def test_luxonis_model_rejects_conflicting_selection():
    with pytest.raises(ValueError, match="'variant' requires 'model'"):
        LuxonisModel(variant="light")
    with pytest.raises(ValueError, match="'cfg' and 'model'"):
        LuxonisModel("foo.yaml", model="detection")


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("detection", ("detection", None)),
        ("detection:v1", ("detection", "1")),
        ("detection:v12", ("detection", "12")),
        ("detection:latest", ("detection", "latest")),
    ],
)
def test_parse_model_spec(model: str, expected: tuple[str, str | None]):
    assert parse_model_spec(model) == expected


@pytest.mark.parametrize(
    "model", ["detection:bad", "detection:2", "detection:v²", "detection:v٣"]
)
def test_parse_model_spec_rejects_malformed(model: str):
    with pytest.raises(ValueError, match="Malformed model spec"):
        parse_model_spec(model)


def test_list_models_cli_command_runs_and_lists_models():
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


def _model_variants() -> list[tuple[str, str | None]]:
    return [
        (model, variant)
        for model in list_predefined_models()
        for variant in list_variants(model)
    ]


@pytest.mark.parametrize(("model", "variant"), _model_variants())
def test_info_runs_for_every_model_variant(
    model: str, variant: str | None, capsys: pytest.CaptureFixture[str]
):
    info(model=model, variant=variant)
    assert capsys.readouterr().out.strip()


def test_configs_use_package_namespace():
    module = importlib.import_module(CONFIGS_PACKAGE)

    assert CONFIGS_PACKAGE == "luxonis_train.configs"
    assert module.__name__ == CONFIGS_PACKAGE
    assert resolve_predefined_config("detection", "light").path.exists()


@pytest.mark.parametrize("opts", [None, ["trainer.epochs", "1"]])
def test_create_model_requires_a_source(opts: list[str] | None):
    with pytest.raises(ValueError, match="No model source given"):
        create_model(None, opts)
