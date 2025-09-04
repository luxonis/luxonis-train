from pathlib import Path

import pytest
from luxonis_ml.data import LuxonisDataset
from tensorboard.backend.event_processing import event_accumulator

from luxonis_train.core import LuxonisModel


@pytest.mark.parametrize(
    ("lr_after_unfreeze", "expected_lrs"),
    [
        (
            0.001,
            {
                "lr-SGD": [(0, 0.000100), (1, 0.002080)],
                "lr-SGD/pg1": [(2, 0.010000), (3, 0.008365)],
                "lr-SGD/pg2": [(2, 0.001000), (3, 0.000837)],
            },
        ),
        (
            0.01,
            {
                "lr-SGD": [(0, 0.000100), (1, 0.002080)],
                "lr-SGD/pg1": [(2, 0.010000), (3, 0.008365)],
                "lr-SGD/pg2": [(2, 0.010000), (3, 0.008365)],
            },
        ),
    ],
)
def test_freezing_parametrized(
    lr_after_unfreeze: float,
    expected_lrs: dict[str, list[tuple[int, float]]],
    coco_dataset: LuxonisDataset,
    image_size: tuple[int, int],
):
    config_file = "configs/segmentation_light_model.yaml"
    opts = {
        "model.predefined_model.params": {
            "head_params": {
                "freezing": {
                    "active": True,
                    "unfreeze_after": 2,
                    "lr_after_unfreeze": lr_after_unfreeze,
                },
            },
        },
        "trainer.epochs": 4,
        "trainer.preprocessing.train_image_size": image_size,
        "loader.params.dataset_name": coco_dataset.identifier,
        "loader.train_view": "val",
        "loader.val_view": "val",
        "loader.test_view": "val",
        "trainer.batch_size": 2,
        "trainer.optimizer": {
            "name": "SGD",
            "params": {
                "lr": 0.01,
                "momentum": 0.9,
                "nesterov": False,
                "weight_decay": 0.0005,
            },
        },
        "trainer.scheduler": {
            "name": "SequentialLR",
            "params": {
                "schedulers": [
                    {
                        "name": "LinearLR",
                        "params": {"start_factor": 0.01},
                    },
                    {
                        "name": "PolynomialLR",
                        "params": {"power": 0.8},
                    },
                ],
                "milestones": [2],
            },
        },
        "trainer.callbacks": [
            {"name": "LearningRateMonitor"},
        ],
    }
    model = LuxonisModel(config_file, opts)
    model.train()
    log_dir = model.lightning_module.logger.experiment["tensorboard"].log_dir

    ea = event_accumulator.EventAccumulator(
        str(Path(log_dir)),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    for tag, expected_seq in expected_lrs.items():
        actual_vals = ea.Scalars(tag)
        assert len(actual_vals) == len(expected_seq), (
            f"{tag}: expected {len(expected_seq)} entries, got {len(actual_vals)}"
        )
        for (exp_step, exp_val), actual in zip(
            expected_seq, actual_vals, strict=True
        ):
            act_step, act_val = actual.step, actual.value
            assert act_step == exp_step, (
                f"{tag}: expected step {exp_step}, got {act_step}"
            )
            assert abs(act_val - exp_val) < 1e-6, (
                f"{tag} @ step {act_step}: expected LR {exp_val:.6f}, got {act_val:.6f}"
            )
