from pathlib import Path

import torch
from luxonis_ml.typing import Params

import luxonis_train as lxt

from ._helpers import config, tiny_head_node


def test_multi_optimizer_training_updates_every_optimizer(
    opts: Params, tmp_path: Path
):
    """A real two-epoch fit with several optimizers must run under
    manual optimization and actually step *every* optimizer.

    `gradient_clip_val` is set on purpose: Lightning refuses automatic
    gradient clipping under manual optimization, so this also exercises
    the manual clipping path in `training_step`.

    Only "at least one parameter per optimizer moved" can be asserted -
    `DummyLoader` yields constant images, so the first convolution's
    weight legitimately receives a zero gradient.
    """
    model = lxt.LuxonisModel(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    }
                )
            ],
            trainer={
                "gradient_clip_val": 1.5,
                "gradient_clip_algorithm": "norm",
            },
        ),
        opts
        | {
            "loader.params.n_classes": 10,
            "trainer.epochs": 2,
            "trainer.accelerator": "cpu",
            # Per-test directory: the tracker's writer threads outlive the
            # test and must not touch the session-shared save directory.
            "tracker.save_directory": str(tmp_path),
            "trainer.n_sanity_val_steps": 0,
        },
        allow_empty_dataset=True,
    )
    module = model.lightning_module
    before = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }

    model.train()

    assert module.automatic_optimization is False
    assert module.trainer.current_epoch == 2

    optimizers = module.optimizers()
    assert isinstance(optimizers, list)
    assert len(optimizers) == 2

    names_by_id = {
        id(parameter): name for name, parameter in module.named_parameters()
    }
    for optimizer in optimizers:
        assert any(
            not torch.equal(
                parameter.detach(), before[names_by_id[id(parameter)]]
            )
            for group in optimizer.param_groups
            for parameter in group["params"]
        ), f"no parameter of {type(optimizer).__name__} was updated"


def test_resuming_past_the_unfreeze_epoch_keeps_training_the_node(
    opts: Params, tmp_path: Path
):
    """Resuming a run whose unfreeze epoch has already passed must leave
    the node trainable and still attached to its optimizer.

    `BaseFinetuning.setup` re-freezes on every fit, and
    `BaseFinetuning.on_fit_start` rebuilds `optimizer.param_groups` from
    the checkpointed metadata. Both have to be undone on the resumed
    run, otherwise the node is either dropped from the optimizer (or the
    checkpointed optimizer state fails to load outright) or it silently
    stays frozen for the rest of training.
    """
    rule: Params = {
        "parameters": [{"module_type": "Linear"}],
        "optimizer": {"name": "SGD", "params": {"lr": 0.05}},
    }
    active = tiny_head_node(None)
    active["alias"] = "ActiveHead"
    active["input_sources"] = ["image"]
    frozen = tiny_head_node(rule)
    frozen["alias"] = "FrozenHead"
    frozen["input_sources"] = ["image"]
    frozen["freezing"] = {"active": True, "unfreeze_after": 1}

    def build(epochs: int, resume: bool) -> lxt.LuxonisModel:
        overrides: Params = {
            "loader.params.n_classes": 10,
            "trainer.epochs": epochs,
            "trainer.accelerator": "cpu",
            "trainer.n_sanity_val_steps": 0,
            "tracker.save_directory": str(tmp_path),
        }
        if resume:
            overrides["trainer.resume_training"] = True
        return lxt.LuxonisModel(
            config(
                [active, frozen],
                trainer={
                    "optimizer": {"name": "Adam", "params": {"lr": 0.01}}
                },
            ),
            opts | overrides,
            allow_empty_dataset=True,
        )

    first = build(2, resume=False)
    first.train()
    checkpoint = max(
        tmp_path.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "state_dict"
    ]

    resumed = build(4, resume=True)
    resumed.train(weights=str(checkpoint))

    module = resumed.lightning_module
    optimized = {
        id(parameter)
        for optimizer in module.trainer.optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    frozen_parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
        if "FrozenHead" in name
    }
    assert frozen_parameters

    for name, parameter in frozen_parameters.items():
        assert parameter.requires_grad, f"'{name}' came back frozen"
        assert id(parameter) in optimized, f"'{name}' belongs to no optimizer"

    # `DummyLoader` yields constant images, so the first convolution's
    # weight keeps a zero gradient - assert the node trained at all.
    assert any(
        not torch.equal(parameter.detach(), saved[name])
        for name, parameter in frozen_parameters.items()
        if name in saved
    )


def test_training_after_unattached_configure_optimizers(
    opts: Params, tmp_path: Path
):
    """Building the optimizers off an unattached module - as
    `build_snapshot` and several tests do - must not poison the
    following fit.

    `configure_optimizers` flips `automatic_optimization` off, but the
    trainer it would reconcile is not reachable yet. `configure_callbacks`
    runs while attached and just before Lightning validates the loop
    configuration, so the trainer-level gradient clipping has to be
    cleared there too; otherwise the fit dies with
    `MisconfigurationException`.
    """
    model = lxt.LuxonisModel(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    }
                )
            ],
            trainer={"gradient_clip_val": 1.5},
        ),
        opts
        | {
            "loader.params.n_classes": 10,
            "trainer.epochs": 1,
            "trainer.accelerator": "cpu",
            "trainer.n_sanity_val_steps": 0,
            "tracker.save_directory": str(tmp_path),
        },
        allow_empty_dataset=True,
    )
    optimizers, _ = model.lightning_module.configure_optimizers()
    assert len(optimizers) == 2
    assert model.lightning_module.automatic_optimization is False

    model.train()

    assert model.lightning_module.trainer.current_epoch == 1
