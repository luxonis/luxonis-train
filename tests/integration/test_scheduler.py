import multiprocessing as mp

import pytest
from luxonis_ml.data import LuxonisDataset

from luxonis_train.config.config import SchedulerConfig
from luxonis_train.core import LuxonisModel


def create_model_config():
    return {
        "trainer": {
            "n_sanity_val_steps": 0,
            "preprocessing": {"train_image_size": [32, 32]},
            "epochs": 2,
            "batch_size": 4,
            "num_workers": mp.cpu_count(),
        },
        "loader": {
            "name": "LuxonisLoaderTorch",
            "train_view": "train",
            "params": {"dataset_name": "coco_test"},
        },
        "model": {
            "name": "detection_light_model",
            "predefined_model": {
                "name": "DetectionModel",
                "params": {
                    "variant": "light",
                },
            },
        },
    }


def sequential_scheduler() -> SchedulerConfig:
    return SchedulerConfig(
        name="SequentialLR",
        params={
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
    )


def cosine_annealing_scheduler() -> SchedulerConfig:
    return SchedulerConfig(
        name="CosineAnnealingLR",
        params={"T_max": 2, "eta_min": 0.001},
    )


@pytest.mark.parametrize(
    "scheduler_config", [sequential_scheduler(), cosine_annealing_scheduler()]
)
def test_scheduler(coco_dataset: LuxonisDataset, scheduler_config):
    config = create_model_config()
    opts = {
        "loader.params.dataset_name": coco_dataset.dataset_name,
        "trainer.scheduler": scheduler_config,
    }
    model = LuxonisModel(config, opts)
    model.train()
