"""Parity between Lightning's automatic optimization and the manual path
that several optimizers switch the trainer into.

The two must agree on *when* learning-rate schedulers step, otherwise a
"with finetuning" run is not comparable to a "without finetuning" one.
"""

from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import pytest
from luxonis_ml.typing import Params
from torch.optim.lr_scheduler import ReduceLROnPlateau

from luxonis_train import LuxonisModel

from ._helpers import config, tiny_head_node

STEP_SCHEDULER: Params = {
    "name": "StepLR",
    "params": {"step_size": 1, "gamma": 0.1},
}
SGD: Params = {"name": "SGD", "params": {"lr": 1.0}}


class ValidationLRRecorder(pl.Callback):
    def __init__(self) -> None:
        self.learning_rates: list[list[float]] = []

    def on_validation_start(
        self, trainer: pl.Trainer, _: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        self.learning_rates.append(
            [
                optimizer.param_groups[0]["lr"]
                for optimizer in trainer.optimizers
            ]
        )


def _validation_learning_rates(
    nodes: list[Params], opts: Params, save_dir: Path
) -> tuple[list[list[float]], bool, int]:
    trainer_cfg = cast(Params, {"optimizer": SGD, "scheduler": STEP_SCHEDULER})
    model = LuxonisModel(
        config(nodes, trainer=trainer_cfg),
        opts
        | {
            "loader.params.n_classes": 10,
            "trainer.epochs": 3,
            "trainer.validation_interval": 1,
            "trainer.n_sanity_val_steps": 0,
            "trainer.accelerator": "cpu",
            # Per-test directory: these are the only unit tests that run a
            # real fit, and the tracker's writer threads outlive them.
            "tracker.save_directory": str(save_dir),
        },
        allow_empty_dataset=True,
    )
    recorder = ValidationLRRecorder()
    cast(Any, model.pl_trainer).callbacks.append(recorder)
    model.train()
    return (
        recorder.learning_rates,
        model.lightning_module.automatic_optimization,
        len(model.pl_trainer.optimizers),
    )


def test_manual_optimization_steps_schedulers_like_lightning(
    opts: Params, tmp_path: Path
):
    """The learning rate seen by the validation loop must be identical
    whether the model trains through Lightning's automatic path (one
    optimizer) or the manual path that finetuning rules trigger.

    Lightning steps epoch-interval, non-plateau schedulers at the end of
    the last training batch, i.e. *before* the validation loop
    (`_TrainingEpochLoop.advance`). `training_step` must step at the
    same point; moving it to `on_train_epoch_end` would make the manual
    path lag one step behind.
    """
    single_lrs, single_automatic, n_single = _validation_learning_rates(
        [tiny_head_node()], opts, tmp_path
    )
    multi_lrs, multi_automatic, n_multi = _validation_learning_rates(
        [
            tiny_head_node(
                [
                    {
                        "parameters": [{"name": "branch1"}],
                        "optimizer": {"name": "Adam", "params": {"lr": 1.0}},
                        "scheduler": STEP_SCHEDULER,
                    }
                ]
            )
        ],
        opts,
        tmp_path,
    )

    assert single_automatic is True
    assert n_single == 1
    assert multi_automatic is False
    assert n_multi == 2

    assert single_lrs == [
        pytest.approx([1e-1]),
        pytest.approx([1e-2]),
        pytest.approx([1e-3]),
    ]
    for epoch_lrs, expected in zip(multi_lrs, single_lrs, strict=True):
        assert epoch_lrs == pytest.approx([expected[0]] * n_multi)


def _plateau_step_epochs(
    nodes: list[Params], opts: Params, save_dir: Path
) -> list[int]:
    trainer_cfg = cast(
        Params,
        {
            "optimizer": SGD,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "params": {"patience": 0},
            },
        },
    )
    model = LuxonisModel(
        config(nodes, trainer=trainer_cfg),
        opts
        | {
            "loader.params.n_classes": 10,
            "trainer.epochs": 4,
            "trainer.validation_interval": 2,
            "trainer.run_validation_after_first_epoch": True,
            "trainer.n_sanity_val_steps": 0,
            "trainer.accelerator": "cpu",
            # Per-test directory: these are the only unit tests that run a
            # real fit, and the tracker's writer threads outlive them.
            "tracker.save_directory": str(save_dir),
        },
        allow_empty_dataset=True,
    )
    epochs: list[int] = []
    original_step = ReduceLROnPlateau.step

    def traced(self: ReduceLROnPlateau, *args, **kwargs) -> None:
        epochs.append(model.lightning_module.current_epoch)
        return original_step(self, *args, **kwargs)

    ReduceLROnPlateau.step = traced  # type: ignore[method-assign]
    try:
        model.train()
    finally:
        ReduceLROnPlateau.step = original_step  # type: ignore[method-assign]
    return sorted(set(epochs))


def test_plateau_scheduler_respects_validation_interval(
    opts: Params, tmp_path: Path
):
    """`ReduceLROnPlateau` is returned with a `frequency` equal to
    `validation_interval`, which Lightning honours on the automatic
    path.

    The manual path drives the scheduler itself and so has to apply the
    same gate. It matters when `run_validation_after_first_epoch`
    temporarily forces validation after epoch 0 while
    `validation_interval` is greater than one: without the gate the
    manual path records an extra observation that the automatic path
    never sees.
    """
    single = _plateau_step_epochs([tiny_head_node()], opts, tmp_path)
    multi = _plateau_step_epochs(
        [
            tiny_head_node(
                [
                    {
                        "parameters": [{"name": "branch1"}],
                        "optimizer": {"name": "Adam", "params": {"lr": 1.0}},
                    }
                ]
            )
        ],
        opts,
        tmp_path,
    )

    assert single == multi
