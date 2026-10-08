from typing import Literal

import pytest
from luxonis_ml.nn_archive.config_building_blocks import PreprocessingBlock
from pydantic import Field

from luxonis_train.config.config import (
    Config,
    NormalizeAugmentationConfig,
    PreprocessingConfig,
)
from luxonis_train.core import LuxonisModel


@pytest.fixture
def model() -> LuxonisModel:
    model = LuxonisModel.__new__(LuxonisModel)
    model.cfg = Config.model_construct()
    model.cfg_preprocessing = PreprocessingConfig()
    return model


@pytest.mark.parametrize(
    ("keep_aspect_ratio", "expected_resize_mode"),
    [(True, "LETTERBOX"), (False, "STRETCH")],
)
def test_archive_preprocessing_resize_mode(
    model: LuxonisModel,
    monkeypatch: pytest.MonkeyPatch,
    keep_aspect_ratio: bool,
    expected_resize_mode: str,
):
    # Exercise the new schema even when the installed luxonis-ml is older.
    monkeypatch.setitem(
        PreprocessingBlock.model_fields, "resize_mode", Field()
    )
    model.cfg_preprocessing.keep_aspect_ratio = keep_aspect_ratio

    preprocessing = model._build_archive_preprocessing()

    assert preprocessing["resize_mode"] == expected_resize_mode


@pytest.mark.parametrize("keep_aspect_ratio", [True, False])
def test_archive_preprocessing_without_resize_mode_support(
    model: LuxonisModel,
    monkeypatch: pytest.MonkeyPatch,
    keep_aspect_ratio: bool,
):
    monkeypatch.delitem(
        PreprocessingBlock.model_fields, "resize_mode", raising=False
    )
    model.cfg_preprocessing.keep_aspect_ratio = keep_aspect_ratio

    preprocessing = model._build_archive_preprocessing()

    assert preprocessing == {
        "mean": [123.675, 116.28, 103.53],
        "scale": [58.395, 57.12, 57.375],
        "dai_type": "RGB888p",
    }


@pytest.mark.parametrize("color_space", ["RGB", "BGR"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("override_mean", [True, False])
@pytest.mark.parametrize("override_scale", [True, False])
def test_archive_preprocessing_normalization(
    model: LuxonisModel,
    color_space: Literal["RGB", "BGR"],
    normalize: bool,
    override_mean: bool,
    override_scale: bool,
):
    model.cfg_preprocessing = PreprocessingConfig(
        color_space=color_space,
        normalize=NormalizeAugmentationConfig(
            active=normalize,
            params={"mean": [0.5, 0.25, 0.125], "std": [0.1, 0.2, 0.4]},
        ),
    )
    if override_mean:
        model.cfg.exporter.mean_values = [1.0, 2.0, 3.0]
    if override_scale:
        model.cfg.exporter.scale_values = [4.0, 5.0, 6.0]

    preprocessing = model._build_archive_preprocessing()

    expected_mean = [127.5, 63.75, 31.875] if normalize else None
    expected_scale = [25.5, 51.0, 102.0] if normalize else None
    assert preprocessing["mean"] == (
        [1.0, 2.0, 3.0] if override_mean else expected_mean
    )
    assert preprocessing["scale"] == (
        [4.0, 5.0, 6.0] if override_scale else expected_scale
    )
    assert preprocessing["dai_type"] == f"{color_space}888p"
