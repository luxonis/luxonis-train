import locale
import os
import subprocess
import sys
from pathlib import Path
from typing import Generator, Tuple

import pytest

os.environ["PYTHONIOENCODING"] = "utf-8"

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

ONNX_PATH = Path("tests/integration/model.onnx")


@pytest.fixture(scope="session", autouse=True)
def clear_files() -> Generator[None, None, None]:
    yield
    ONNX_PATH.unlink(missing_ok=True)


def run_command(command: str) -> Tuple[str, str, int]:
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding=locale.getpreferredencoding(),
        errors="replace",
    )
    return result.stdout, result.stderr, result.returncode


@pytest.mark.parametrize(
    "command",
    [
        "luxonis_train train --config tests/configs/cli_commands.yaml",
        "luxonis_train test --config tests/configs/cli_commands.yaml --view val",
        f"luxonis_train export --config tests/configs/cli_commands.yaml --save-path {ONNX_PATH}",
        f"luxonis_train archive --config tests/configs/cli_commands.yaml --executable {ONNX_PATH}",
        "luxonis_train --version",
    ],
)
def test_cli_command_success(command: str) -> None:
    stdout, stderr, exit_code = run_command(command)
    assert exit_code == 0, f"Error: {stderr}"
    assert "Error" not in stderr


@pytest.mark.parametrize(
    "command",
    [
        "luxonis_train train --config nonexistent.yaml",
        "luxonis_train test --config nonexistent.yaml --view invalid",
        "luxonis_train export --config nonexistent.yaml",
        "luxonis_train inspect --config nonexistent.yaml --view train --size-multiplier -1.0",
        "luxonis_train archive --config nonexistent.yaml",
    ],
)
def test_cli_command_failure(command: str) -> None:
    stdout, stderr, exit_code = run_command(command)
    assert exit_code != 0, f"Expected failure but got: {stderr}"
    assert "Error" in stderr or stderr.strip()
