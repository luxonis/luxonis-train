import os
import subprocess
from pathlib import Path

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.utils import environ


def test_source(work_dir: Path, coco_dataset: LuxonisDataset):
    with open(work_dir / "source_1.py", "w") as f:
        f.write("print('sourcing 1')")

    with open(work_dir / "source_2.py", "w") as f:
        f.write("print('sourcing 2')")

    result = subprocess.run(
        [  # noqa: S607
            "python",
            "-m",
            "luxonis_train",
            "--source",
            str(work_dir / "source_1.py"),
            "test",
            "--source",
            str(work_dir / "source_2.py"),
            "--config",
            "tests/configs/config_simple.yaml",
            "loader.params.dataset_name",
            coco_dataset.identifier,
        ],
        capture_output=True,
        text=True,
        env=os.environ | {"LUXONISML_BASE_PATH": environ.LUXONISML_BASE_PATH},
    )

    assert result.returncode == 0, (
        f"Command failed with error: {result.stderr}"
    )

    assert "sourcing 1" in result.stdout
    assert "sourcing 2" in result.stdout
