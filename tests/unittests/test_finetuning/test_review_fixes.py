"""Regression tests for the invariants the parameter-groups rework
initially dropped.

Each test reproduces a concrete way training used to go silently wrong
and asserts the restored behaviour.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from lightning.pytorch.callbacks import BaseFinetuning
from luxonis_ml.typing import Params
from torch import Size, nn
from torch.optim import SGD, Adagrad, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import luxonis_train as lxt
from luxonis_train.callbacks import TrainingManager
from luxonis_train.config import Config
from luxonis_train.config.config import SchedulerConfig, SequentialLRParams
from luxonis_train.lightning.utils import get_gradient_accumulation_schedule

from ._helpers import config, node, tiny_head_node

TRIPLE_LR = {"name": "TripleLRSGDStrategy", "params": {"lr": 0.02}}


def build(cfg: Params, opts: Params) -> lxt.LuxonisModel:
    return lxt.LuxonisModel(
        deepcopy(cfg),
        opts | {"loader.params.n_classes": 10},
        allow_empty_dataset=True,
    )


def frozen_node(name: str, unfreeze_after: int = 1, **extra: Any) -> Params:
    node_cfg = node(name)
    node_cfg["freezing"] = {"active": True, "unfreeze_after": unfreeze_after}
    node_cfg.update(extra)
    return node_cfg


def optimized_ids(optimizers: list[Optimizer]) -> set[int]:
    return {
        id(parameter)
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def trainable_ids(module: nn.Module) -> set[int]:
    return {
        id(parameter)
        for parameter in module.parameters()
        if parameter.requires_grad
    }


def unfreeze(
    module: "lxt.LuxonisLightningModule",
    optimizers: list[Optimizer],
    epoch: int = 1,
) -> TrainingManager:
    """Run the freeze/unfreeze cycle the way Lightning drives it."""
    manager = TrainingManager()
    manager.finetune_function(module, epoch, optimizers[0])
    return manager


def test_unfrozen_parameters_are_never_orphaned_with_a_strategy(
    opts: Params,
):
    """A frozen node with no ``finetuning`` entry of its own used to end
    up in no optimizer at all once a training strategy and any
    ``finetuning`` entry were combined.

    The strategy skips frozen parameters and only claims what is left
    over, and the strategy path builds node optimizers with
    ``include_default=False``, so nothing registered an unfreeze target
    for that node. Its weights then stayed at their initial values for
    the whole run, silently.
    """
    model = build(
        config(
            [
                frozen_node("Backbone"),
                node("Neck"),
                node(
                    "Head",
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"name": "SGD", "params": {"lr": 0.01}},
                    },
                ),
            ],
            trainer={"training_strategy": TRIPLE_LR},
        ),
        opts,
    )
    module = model.lightning_module

    manager = TrainingManager()
    manager.freeze_before_training(module)
    optimizers, _ = module.configure_optimizers()
    optimizers = list(optimizers)

    backbone = module.nodes["Backbone"].module
    assert not any(p.requires_grad for p in backbone.parameters())

    manager.finetune_function(module, 1, optimizers[0])

    assert all(p.requires_grad for p in backbone.parameters())
    orphaned = trainable_ids(module) - optimized_ids(optimizers)
    assert orphaned == set()


def test_unfreezing_adds_a_new_parameter_group(opts: Params):
    """``BaseFinetuning._store`` only records parameter groups appended
    past the ones it already knows about.

    Extending an existing group in place therefore left the checkpoint
    metadata stale, and resuming a run would drop the newly unfrozen
    parameters from the optimizer entirely.
    """
    head = tiny_head_node(
        {
            "parameters": [{"module_type": "Linear"}],
            "optimizer": {"name": "SGD", "params": {"lr": 0.02}},
        }
    )
    head["freezing"] = {"active": True, "unfreeze_after": 1}
    model = build(config([head]), opts)
    module = model.lightning_module

    manager = TrainingManager()
    manager.freeze_before_training(module)
    optimizers, _ = module.configure_optimizers()
    optimizers = list(optimizers)

    before = [len(optimizer.param_groups) for optimizer in optimizers]
    manager.finetune_function(module, 1, optimizers[0])
    after = [len(optimizer.param_groups) for optimizer in optimizers]

    assert sum(after) > sum(before)
    assert trainable_ids(module) <= optimized_ids(optimizers)


def test_unfreezing_survives_param_groups_being_rebuilt(opts: Params):
    """On resume, ``BaseFinetuning.on_fit_start`` replaces
    ``optimizer.param_groups`` with freshly built dicts.

    Holding on to a group dict captured at build time would leave the
    unfrozen parameters in a detached dict that no optimizer reads.
    """
    head = tiny_head_node(
        {
            "parameters": [{"module_type": "Linear"}],
            "optimizer": {"name": "SGD", "params": {"lr": 0.02}},
        }
    )
    head["freezing"] = {"active": True, "unfreeze_after": 1}
    model = build(config([head]), opts)
    module = model.lightning_module

    manager = TrainingManager()
    manager.freeze_before_training(module)
    optimizers, _ = module.configure_optimizers()
    optimizers = list(optimizers)

    # What `on_fit_start` does when restarting: rebuild every group dict
    # from the stored metadata.
    named_parameters = dict(module.named_parameters())
    for optimizer in optimizers:
        mapping = {p: n for n, p in named_parameters.items()}
        metadata = BaseFinetuning._apply_mapping_to_param_groups(
            optimizer.param_groups, mapping
        )
        optimizer.param_groups = BaseFinetuning._apply_mapping_to_param_groups(
            metadata, named_parameters
        )

    manager.finetune_function(module, 1, optimizers[0])

    assert trainable_ids(module) <= optimized_ids(optimizers)


def test_unfreeze_uses_the_current_learning_rate(opts: Params):
    """Without an explicit ``lr_after_unfreeze`` the parameters must
    join at the rate the scheduler has decayed to.

    Using the configured initial rate instead would hand freshly
    unfrozen pretrained weights a learning-rate jump exactly at the
    unfreeze epoch. The frozen node here has no ``finetuning`` entry, so
    its parameters form their own group with no trainable member, which
    is the case that took the configured rate verbatim.
    """
    model = build(
        config(
            [frozen_node("Backbone"), node("Neck"), node("Head")],
            trainer={"optimizer": {"name": "SGD", "params": {"lr": 0.02}}},
        ),
        opts,
    )
    module = model.lightning_module

    manager = TrainingManager()
    manager.freeze_before_training(module)
    optimizers, _ = module.configure_optimizers()
    optimizers = list(optimizers)

    # Stand in for a scheduler having decayed the rate by the time the
    # node is unfrozen.
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = 0.01
    group_counts = [len(optimizer.param_groups) for optimizer in optimizers]

    manager.finetune_function(module, 1, optimizers[0])

    added = [
        group
        for optimizer, count in zip(optimizers, group_counts, strict=True)
        for group in optimizer.param_groups[count:]
    ]
    assert added, "unfrozen parameters must be added as a new group"
    for group in added:
        assert group["lr"] == pytest.approx(0.01)


def test_explicit_lr_after_unfreeze_still_wins(opts: Params):
    head = tiny_head_node()
    head["freezing"] = {
        "active": True,
        "unfreeze_after": 1,
        "lr_after_unfreeze": 0.007,
    }
    model = build(config([head]), opts)
    module = model.lightning_module

    manager = TrainingManager()
    manager.freeze_before_training(module)
    optimizers, _ = module.configure_optimizers()
    optimizers = list(optimizers)
    before = len(optimizers[0].param_groups)

    manager.finetune_function(module, 1, optimizers[0])

    added = optimizers[0].param_groups[before:]
    assert added
    assert all(group["lr"] == pytest.approx(0.007) for group in added)


def test_constructor_only_optimizer_arguments_are_applied(opts: Params):
    """Hyperparameters used to travel only inside the per-group dicts,
    so arguments a torch optimizer consumes in ``__init__`` were
    silently dropped.

    ``Adagrad`` seeds its accumulator state from
    ``initial_accumulator_value`` in the constructor and ignores the key
    when it appears on a parameter group.
    """
    model = build(
        config(
            [tiny_head_node()],
            trainer={
                "optimizer": {
                    "name": "Adagrad",
                    "params": {
                        "lr": 0.01,
                        "initial_accumulator_value": 0.7,
                    },
                }
            },
        ),
        opts,
    )
    optimizers, _ = model.lightning_module.configure_optimizers()
    optimizer = next(iter(optimizers))
    assert isinstance(optimizer, Adagrad)

    sums = [
        optimizer.state[parameter]["sum"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert sums
    assert all(
        torch.allclose(value, torch.full_like(value, 0.7)) for value in sums
    )


def test_invalid_optimizer_hyperparameters_are_rejected(opts: Params):
    """Passing the options to the constructor also restores torch's own
    validation, which never ran while they lived on group dicts only.
    """
    model_config = config(
        [tiny_head_node()],
        trainer={
            "optimizer": {
                "name": "SGD",
                "params": {"lr": 0.01, "nesterov": True, "momentum": 0},
            }
        },
    )
    model = build(model_config, opts)
    with pytest.raises(ValueError, match=r"[Nn]esterov"):
        model.lightning_module.configure_optimizers()


def test_no_trainable_parameters_raises_a_clear_error(opts: Params):
    """An empty optimizer list used to reach Lightning's training loop
    and die there with ``IndexError: list index out of range``.
    """
    model = build(config([tiny_head_node()]), opts)
    module = model.lightning_module
    for parameter in module.parameters():
        parameter.requires_grad = False

    with pytest.raises(ValueError, match="no parameter of the model"):
        module.configure_optimizers()


def test_automatic_optimization_is_restored_between_runs(opts: Params):
    """``automatic_optimization`` lives on the module, so leaving it at
    ``False`` leaked into later ``fit`` calls and made Lightning reject
    ``gradient_clip_val`` that the user never changed.
    """
    model = build(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"name": "SGD"},
                    }
                )
            ]
        ),
        opts,
    )
    module = model.lightning_module

    optimizers, _ = module.configure_optimizers()
    assert len(list(optimizers)) > 1
    assert module.automatic_optimization is False

    # A second `configure_optimizers` that yields a single optimizer
    # must hand the flag back.
    module.nodes["Head"].finetuning = []
    optimizers, _ = module.configure_optimizers()
    assert len(list(optimizers)) == 1
    assert module.automatic_optimization is True


def test_expected_optimizer_count_accounts_for_freezing(opts: Params):
    """``configure_callbacks`` runs before ``TrainingManager`` freezes
    anything, so counting optimizers against the live ``requires_grad``
    flags overestimated and silently disabled gradient accumulation.
    """
    model = build(
        config(
            [
                # Every unfrozen parameter is claimed by a rule, so once
                # the Backbone is frozen the strategy has nothing left
                # and contributes no optimizer of its own.
                frozen_node("Backbone"),
                node(
                    "Neck",
                    {
                        "parameters": [{"name": ".*"}],
                        "optimizer": {"name": "SGD", "params": {"lr": 0.01}},
                    },
                ),
                node(
                    "Head",
                    {
                        "parameters": [{"name": ".*"}],
                        "optimizer": {"name": "SGD", "params": {"lr": 0.01}},
                    },
                ),
            ],
            trainer={
                "training_strategy": TRIPLE_LR,
                "accumulate_grad_batches": 4,
            },
        ),
        opts,
    )
    module = model.lightning_module

    estimated = module._expected_optimizer_count()

    TrainingManager().freeze_before_training(module)
    optimizers, _ = module.configure_optimizers()

    assert estimated == len(list(optimizers))


def test_gradient_accumulation_is_applied_under_manual_optimization(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """Lightning refuses to accumulate gradients itself under manual
    optimization, so the configured factor has to be honoured in the
    training step.

    Dropping it silently was actively harmful for predefined models,
    whose loss weights ``smart_cfg_auto_populate`` has already scaled by
    the very same factor.
    """
    model = build(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"name": "SGD"},
                    }
                )
            ],
            trainer={"accumulate_grad_batches": 3},
        ),
        opts,
    )
    module = model.lightning_module
    optimizers, _ = module.configure_optimizers()
    optimizers = list(optimizers)
    assert module.automatic_optimization is False
    assert module._accumulation_factor() == 3

    backward_losses: list[float] = []
    steps: list[int] = []

    monkeypatch.setattr(
        module,
        "compute_training_loss",
        lambda _batch: torch.tensor(3.0, requires_grad=True),
    )
    monkeypatch.setattr(
        module, "optimizers", lambda *_a, **_k: list(optimizers)
    )
    monkeypatch.setattr(module, "lr_schedulers", list)
    monkeypatch.setattr(
        module,
        "manual_backward",
        lambda loss: backward_losses.append(float(loss)),
    )
    for index, optimizer in enumerate(optimizers):
        monkeypatch.setattr(
            optimizer, "step", lambda index=index: steps.append(index)
        )
        monkeypatch.setattr(optimizer, "zero_grad", lambda: None)
    module._trainer = SimpleNamespace(  # type: ignore[assignment]
        is_last_batch=False, current_epoch=0
    )

    for batch_idx in range(6):
        module.training_step((torch.empty(0), {}), batch_idx=batch_idx)

    # The loss is normalized the same way automatic optimization does.
    assert backward_losses == [pytest.approx(1.0)] * 6
    # Two accumulation windows over six batches, each optimizer stepping
    # once per window.
    assert len(steps) == 2 * len(optimizers)


def test_gradient_accumulation_schedule_is_read_from_the_callback():
    cfg = Config.get_config(
        config(
            [tiny_head_node()],
            trainer={
                "callbacks": [
                    {
                        "name": "GradientAccumulationScheduler",
                        "params": {"scheduling": {0: 1, 2: 4}},
                    }
                ]
            },
        )
    )
    assert get_gradient_accumulation_schedule(cfg) == {0: 1, 2: 4}


def test_accumulate_grad_batches_is_read_when_no_callback_is_set():
    cfg = Config.get_config(
        config([tiny_head_node()], trainer={"accumulate_grad_batches": 8})
    )
    assert get_gradient_accumulation_schedule(cfg) == {0: 8}


def test_global_gradient_clipping_scales_with_the_whole_model(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """Clipping each optimizer separately let the total applied norm
    grow with the number of optimizers.

    With two optimizers each holding a unit-norm gradient, per-optimizer
    clipping to 1.0 would be a no-op while a global clip has to scale
    both down.
    """
    model = build(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"name": "SGD"},
                    }
                )
            ],
            trainer={"gradient_clip_val": 1.0},
        ),
        opts,
    )
    module = model.lightning_module
    optimizers, _ = module.configure_optimizers()
    assert len(list(optimizers)) > 1

    parameters = [p for p in module.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    # Two parameters carrying a unit-norm gradient each: total norm √2.
    parameters[0].grad.view(-1)[0] = 1.0
    parameters[-1].grad.view(-1)[0] = 1.0

    module._clip_gradients_globally()

    total_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(p.grad.detach())
                for p in parameters
                if p.grad is not None
            ]
        )
    )
    assert float(total_norm) == pytest.approx(1.0, abs=1e-5)


def test_plateau_scheduler_reduces_the_loss_across_ranks(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """The loss accumulator is rank-local.

    Stepping ``ReduceLROnPlateau`` with it would let ranks disagree
    about when to decay, and learning rates are optimizer state DDP
    never re-synchronizes.
    """
    model = build(config([tiny_head_node()]), opts)
    module = model.lightning_module
    module.automatic_optimization = False

    reduced: list[float] = []

    def fake_reduce(tensor: torch.Tensor, reduce_op: str = "mean") -> Any:
        reduced.append(float(tensor))
        assert reduce_op == "mean"
        return torch.tensor(0.25)

    module._trainer = SimpleNamespace(  # type: ignore[assignment]
        world_size=2,
        strategy=SimpleNamespace(reduce=fake_reduce),
    )

    optimizer = SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    plateau = ReduceLROnPlateau(optimizer, mode="min")
    stepped: list[float] = []
    monkeypatch.setattr(plateau, "step", stepped.append)
    monkeypatch.setattr(module, "lr_schedulers", lambda: plateau)

    module._step_reduce_lr_on_plateau_schedulers(
        loss=0.42, metrics={"Head": {"Accuracy": 0.9}}
    )

    assert reduced == [pytest.approx(0.42)]
    assert stepped == [pytest.approx(0.25)]


def test_strategy_base_weight_decay_skips_biases_and_batchnorm(
    opts: Params,
):
    """``TripleLRSGD`` deliberately decays only regular weights.

    A ``finetuning`` rule inheriting ``weight_decay`` from the
    strategy's base config used to apply it to every parameter it
    matched, turning on exactly the regularisation the strategy avoids.
    """
    model = build(
        config(
            [tiny_head_node({"parameters": [{"name": ".*"}]})],
            trainer={"training_strategy": TRIPLE_LR},
        ),
        opts,
    )
    module = model.lightning_module
    optimizers, _ = module.configure_optimizers()
    inherited = next(iter(optimizers))

    names = {id(p): name for name, p in module.named_parameters()}
    decay_by_name = {
        names[id(parameter)]: group["weight_decay"]
        for group in inherited.param_groups
        for parameter in group["params"]
    }
    assert decay_by_name

    biases = [n for n in decay_by_name if n.endswith(".bias")]
    weights = [n for n in decay_by_name if n.endswith(".weight")]
    assert biases
    assert weights

    for name in biases:
        assert decay_by_name[name] == pytest.approx(0.0), name
    for name in weights:
        assert decay_by_name[name] == pytest.approx(0.0005), name


def test_explicit_weight_decay_on_a_rule_is_left_alone(opts: Params):
    """Leave a ``weight_decay`` written on the rule itself alone.

    The split only applies to one inherited from the strategy; one the
    user wrote expresses explicit intent.
    """
    model = build(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"params": {"weight_decay": 0.0004}},
                    }
                )
            ],
            trainer={"training_strategy": TRIPLE_LR},
        ),
        opts,
    )
    optimizers, _ = model.lightning_module.configure_optimizers()
    groups = [
        group
        for group in next(iter(optimizers)).param_groups
        if group["params"]
    ]
    assert groups
    assert all(
        group["weight_decay"] == pytest.approx(0.0004) for group in groups
    )


def test_strategy_warmup_covers_finetuning_optimizers(opts: Params):
    """``update_parameters`` only touched the strategy's own optimizer,
    so parameters claimed by ``finetuning`` rules started at full
    learning rate while the rest of the model was still warming up.
    """
    model = build(
        config(
            [tiny_head_node({"parameters": [{"module_type": "Linear"}]})],
            trainer={"training_strategy": TRIPLE_LR},
        ),
        opts,
    )
    module = model.lightning_module
    optimizers, _ = module.configure_optimizers()
    finetuning_optimizer = next(iter(optimizers))

    configured = [group["lr"] for group in finetuning_optimizer.param_groups]
    assert all(lr > 0 for lr in configured)

    assert module.training_strategy is not None
    module.training_strategy.update_parameters()

    warmed = [group["lr"] for group in finetuning_optimizer.param_groups]
    assert all(lr < configured[i] for i, lr in enumerate(warmed))


def test_sequential_lr_scheduler_requires_a_name():
    """``SchedulerConfig.name`` gained a default, which turned a missing
    name on a nested ``SequentialLR`` entry into a silent ``ConstantLR``
    instead of a validation error.
    """
    with pytest.raises(ValueError, match="`name`"):
        SequentialLRParams(
            schedulers=[
                {"name": "LinearLR", "params": {"total_iters": 5}},  # type: ignore[list-item]
                {"params": {"T_max": 100}},  # type: ignore[list-item]
            ],
            milestones=[5],
        )


def test_scheduler_params_without_a_name_are_rejected():
    with pytest.raises(ValueError, match="without a `name`"):
        SchedulerConfig(params={"T_max": 100})

    # The default stays usable when nothing is configured.
    assert SchedulerConfig().name == "ConstantLR"
    assert SchedulerConfig(name="StepLR", params={"step_size": 1}).params


def test_ocr_head_still_accepts_the_removed_fc_decay():
    """``fc_decay`` was a public node parameter.

    Removing it outright turned every existing OCR config into a
    ``TypeError`` at model construction, since nodes take no
    ``**kwargs`` catch-all. It is accepted again, ignored, and reported.
    """
    from luxonis_train.nodes.heads.ocr_ctc_head import OCRCTCHead

    head = OCRCTCHead(
        alphabet=["a", "b", "c"],
        fc_decay=0.0004,
        input_shapes=[{"features": [Size((4, 8, 1, 16))]}],
        original_in_shape=Size((3, 32, 128)),
        task_name="ocr",
    )
    assert isinstance(head, OCRCTCHead)
    # Ignored, not stored: it never applied any regularization.
    assert not hasattr(head, "fc_decay")
