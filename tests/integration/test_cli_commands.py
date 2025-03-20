import os
from collections.abc import Callable
from pathlib import Path

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Kwargs
from luxonis_ml.utils import environ

from luxonis_train.__main__ import (
    archive,
    export,
    inspect,
    train,
)
from luxonis_train.__main__ import test as _test

ONNX_PATH = Path("tests/integration/model.onnx")


@pytest.fixture(scope="session", autouse=True)
def prepare():
    os.environ["LUXONISML_BASE_PATH"] = str(environ.LUXONISML_BASE_PATH)
    yield
    ONNX_PATH.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("command", "kwargs"),
    [
        (train, {"config": "tests/configs/config_simple.yaml"}),
        (
            _test,
            {
                "config": "tests/configs/config_simple.yaml",
                "view": "val",
            },
        ),
        (
            export,
            {
                "config": "tests/configs/config_simple.yaml",
                "save_path": ONNX_PATH,
            },
        ),
        (
            archive,
            {
                "config": "tests/configs/config_simple.yaml",
                "executable": ONNX_PATH,
            },
        ),
    ],
)
def test_cli_command_success(
    command: Callable, kwargs: Kwargs, coco_dataset: LuxonisDataset
) -> None:
    command(["loader.params.dataset_name", coco_dataset.identifier], **kwargs)


@pytest.mark.parametrize(
    ("command", "kwargs"),
    [
        (train, {"config": "nonexistent.yaml"}),
        (
            _test,
            {"config": "tests/configs/config_simple.yaml", "view": "invalid"},
        ),
        (export, {"config": "nonexistent.yaml"}),
        (
            inspect,
            {
                "config": "nonexistent.yaml",
                "view": "train",
                "size_multiplier": -1.0,
            },
        ),
        (archive, {"config": "nonexistent.yaml"}),
    ],
)
def test_cli_command_failure(
    command: Callable, kwargs: Kwargs, coco_dataset: LuxonisDataset
) -> None:
    with pytest.raises(Exception):  # noqa: B017 PT011
        command(
            ["loader.params.dataset_name", coco_dataset.identifier],
            **kwargs,
        )
