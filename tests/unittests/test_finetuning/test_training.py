from pathlib import Path

import pytest
import torch
from luxonis_ml.typing import Params

import luxonis_train as lxt
from luxonis_train.lightning.training_plan import unwrap_optimizers

from ._helpers import config, tiny_head_node


def test_multi_optimizer_training_updates_every_optimizer(
    opts: Params, tmp_path: Path
):
    """A real two-epoch fit with two optimizer configurations must stay
    in automatic optimization (one composite optimizer) and actually
    step *every* inner optimizer, with trainer-level gradient clipping
    active.

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

    assert module.automatic_optimization is True
    assert module.trainer.current_epoch == 2

    optimizers = unwrap_optimizers(module.trainer.optimizers)
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

    Several optimizer configurations stay in automatic optimization
    (one composite optimizer), so trainer-level gradient clipping
    remains valid and untouched.
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
    assert len(optimizers) == 1
    assert len(unwrap_optimizers(list(optimizers))) == 2
    assert model.lightning_module.automatic_optimization is True

    model.train()

    assert model.lightning_module.trainer.current_epoch == 1
    # automatic optimization keeps Lightning's own clipping valid
    assert model.pl_trainer.gradient_clip_val == 1.5


def test_strategy_with_freezing_trains_unfrozen_node(
    opts: Params, tmp_path: Path
):
    """A training strategy combined with a finetuning rule and a frozen
    node must still train the node after it unfreezes.

    This is the regression test for the silent failure where the frozen
    node's parameters ended up in no optimizer at all (the strategy
    filtered them out at build time and the exclusion-set contract could
    not adopt them later). The total partition guarantees they sit in a
    group from the start.
    """
    active = tiny_head_node({"parameters": [{"module_type": "Linear"}]})
    active["alias"] = "ActiveHead"
    active["input_sources"] = ["image"]
    frozen = tiny_head_node(None)
    frozen["alias"] = "FrozenHead"
    frozen["input_sources"] = ["image"]
    frozen["freezing"] = {"active": True, "unfreeze_after": 1}

    model = lxt.LuxonisModel(
        config(
            [active, frozen],
            trainer={
                "training_strategy": {
                    "name": "TripleLRSGDStrategy",
                    "params": {"warmup_epochs": 0},
                }
            },
        ),
        opts
        | {
            "loader.params.n_classes": 10,
            "trainer.epochs": 3,
            "trainer.accelerator": "cpu",
            "trainer.n_sanity_val_steps": 0,
            "tracker.save_directory": str(tmp_path),
        },
        allow_empty_dataset=True,
    )
    module = model.lightning_module
    before = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }

    model.train()

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
        assert parameter.requires_grad, f"'{name}' is still frozen"
        assert id(parameter) in optimized, f"'{name}' has no optimizer"
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in frozen_parameters.items()
    ), "the unfrozen node never trained"


def test_gradient_accumulation_with_multiple_inner_optimizers(
    opts: Params, tmp_path: Path
):
    """Gradient accumulation works with several optimizer
    configurations: the composite steps once per accumulation window.

    Previously any finetuning topology with more than one optimizer
    switched to manual optimization and silently ignored
    ``accumulate_grad_batches``.
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
            trainer={"accumulate_grad_batches": 2},
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
    module = model.lightning_module

    model.train()

    assert module.automatic_optimization is True
    assert len(unwrap_optimizers(module.trainer.optimizers)) == 2
    train_dataloader = model.pl_trainer.train_dataloader
    assert train_dataloader is not None
    n_batches = len(train_dataloader)
    assert module.trainer.global_step == n_batches // 2


def test_resume_before_unfreeze_applies_lr_after_unfreeze_once(
    opts: Params, tmp_path: Path
):
    """Resuming from a checkpoint written *before* the unfreeze epoch
    must fire the unfreeze transition in the resumed run and apply
    ``lr_after_unfreeze`` as the group's base learning rate.
    """
    active = tiny_head_node(None)
    active["alias"] = "ActiveHead"
    active["input_sources"] = ["image"]
    frozen = tiny_head_node(None)
    frozen["alias"] = "FrozenHead"
    frozen["input_sources"] = ["image"]
    frozen["freezing"] = {
        "active": True,
        "unfreeze_after": 2,
        "lr_after_unfreeze": 0.5,
    }

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
            config([active, frozen]),
            opts | overrides,
            allow_empty_dataset=True,
        )

    first = build(1, resume=False)
    first.train()
    checkpoint = max(
        tmp_path.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime
    )

    resumed = build(4, resume=True)
    resumed.train(weights=str(checkpoint))

    module = resumed.lightning_module
    runtime = module.training_plan
    assert runtime is not None
    (handle,) = runtime.handles_for_node("FrozenHead")
    group = runtime.group(handle)
    assert group["lr"] == pytest.approx(0.5)
    assert group["initial_lr"] == pytest.approx(0.5)
    assert all(parameter.requires_grad for parameter in module.parameters())


def test_mixed_precision_composite_smoke(opts: Params, tmp_path: Path):
    """One fit under ``16-mixed`` precision with two inner optimizers -
    the composite must be indistinguishable from a plain optimizer to
    the GradScaler-driven precision plugin.
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
        ),
        opts
        | {
            "loader.params.n_classes": 10,
            "trainer.epochs": 1,
            "trainer.accelerator": "cpu",
            "trainer.precision": "16-mixed",
            "trainer.n_sanity_val_steps": 0,
            "tracker.save_directory": str(tmp_path),
        },
        allow_empty_dataset=True,
    )

    model.train()

    module = model.lightning_module
    assert module.automatic_optimization is True
    assert module.trainer.current_epoch == 1
    assert len(unwrap_optimizers(module.trainer.optimizers)) == 2
