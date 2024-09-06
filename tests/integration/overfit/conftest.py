import multiprocessing as mp
import os
from typing import Any

import pytest

LUXONIS_TRAIN_OVERFIT = os.getenv("LUXONIS_TRAIN_OVERFIT") or False

if LUXONIS_TRAIN_OVERFIT:
    EPOCHS = 200
else:
    EPOCHS = 1


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "tracker": {
            "save_directory": "tests/integration/save-directory",
        },
        "loader": {
            "train_view": "val",
            "params": {
                "dataset_name": "_ParkingLot",
            },
        },
        "trainer": {
            "batch_size": 4,
            "epochs": EPOCHS,
            "num_workers": mp.cpu_count(),
            "validation_interval": EPOCHS,
            "save_top_k": 0,
            "preprocessing": {
                "train_image_size": [256, 320],
                "keep_aspect_ratio": False,
                "normalize": {"active": True},
            },
            "callbacks": [
                {"name": "ExportOnTrainEnd"},
                {"name": "ArchiveOnTrainEnd"},
            ],
        },
    }
