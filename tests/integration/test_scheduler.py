from typing import Any

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params, ParamValue

from luxonis_train.core import LuxonisModel


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
    "scheduler_config",
    [
        sequential_scheduler(),
        cosine_annealing_scheduler(),
        sequential_with_reduce_on_plateau_scheduler(),
    ],
)
def test_scheduler(
    opts: Params, coco_dataset: LuxonisDataset, scheduler_config: ParamValue
):
    opts |= {
        "loader.params.dataset_name": coco_dataset.dataset_name,
        "trainer.scheduler": scheduler_config,
    }
    model = LuxonisModel("configs/detection_light_model.yaml", opts)
    model.train()
