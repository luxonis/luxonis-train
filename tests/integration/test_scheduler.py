import multiprocessing as mp
from typing import Any

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def create_model_config() -> dict[str, Any]:
    return {
        "trainer": {
            "n_sanity_val_steps": 0,
            "preprocessing": {"train_image_size": [32, 32]},
            "epochs": 2,
            "batch_size": 2,
            "n_workers": mp.cpu_count(),
            "validation_interval": 1,
        },
        "loader": {
            "name": "LuxonisLoaderTorch",
            "train_view": "val",
            "val_view": "val",
            "test_view": "val",
            "params": {"dataset_name": "coco_test"},
        },
        "model": {
            "name": "detection_light_model",
            "predefined_model": {
                "name": "DetectionModel",
                "params": {"variant": "light"},
            },
        },
    }


def sequential_scheduler() -> dict[str, Any]:
    return {
        "name": "SequentialLR",
        "params": {
            "schedulers": [
                {
                    "name": "LinearLR",
                    "params": {"start_factor": 0.1, "total_iters": 10},
                },
                {
                    "name": "CosineAnnealingLR",
                    "params": {"T_max": 1, "eta_min": 0.01},
                },
            ],
            "milestones": [1],
        },
    }


def cosine_annealing_scheduler() -> dict[str, Any]:
    return {
        "name": "CosineAnnealingLR",
        "params": {"T_max": 2, "eta_min": 0.001},
    }


def sequential_with_reduce_on_plateau_scheduler() -> dict[str, Any]:
    return {
        "name": "ReduceLROnPlateau",
        "params": {"mode": "max", "factor": 0.1, "patience": 2},
    }


@pytest.mark.parametrize(
    "scheduler_config", [sequential_scheduler(), cosine_annealing_scheduler()]
)
def test_scheduler(coco_dataset: LuxonisDataset, scheduler_config: Params):
    config = create_model_config()
    opts = {
        "loader.params.dataset_name": coco_dataset.dataset_name,
        "trainer.scheduler": scheduler_config,
    }
    model = LuxonisModel(config, opts)
    model.train()
