import locale
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ["PYTHONIOENCODING"] = "utf-8"

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ONNX_PATH = Path("tests/integration/model.onnx")
ARCHIVE_PATH = Path("tests/integration/model.nn")


@pytest.fixture(scope="session", autouse=True)
def clear_files():
    yield
    ONNX_PATH.unlink(missing_ok=True)
    ARCHIVE_PATH.unlink(missing_ok=True)


def run_command(command):
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
def test_cli_command_success(command):
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
def test_cli_command_failure(command):
    stdout, stderr, exit_code = run_command(command)
    assert exit_code != 0, f"Expected failure but got: {stderr}"
    assert "Error" in stderr or stderr.strip()
