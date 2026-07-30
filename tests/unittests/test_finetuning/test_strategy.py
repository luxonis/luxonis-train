from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerTypeUnion,
)
from luxonis_ml.typing import Params
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
)
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.config.config import OptimizerConfig, SchedulerConfig
from luxonis_train.lightning import luxonis_lightning
from luxonis_train.lightning.utils import MainMetric
from luxonis_train.strategies.base_strategy import BaseTrainingStrategy

from ._helpers import (
    OptimizerSnapshot,
    assert_all_trainable_parameters_assigned,
    assert_group_options,
    assert_no_duplicate_parameters,
    build_snapshot,
    config,
    find_group,
    ids_for_names,
    matching_names,
    names_for_ids,
    optimizer_group_names,
    optimizer_names,
    scheduler,
    tiny_head_node,
    trainable_parameter_names,
)


class CapturingFinetuningStrategy(BaseTrainingStrategy):
    def __init__(
        self,
        pl_module: "lxt.LuxonisLightningModule",
        lr: float = 0.02,
        base_lr: float = 0.031,
        base_step_size: int = 9,
    ):
        self.pl_module = pl_module
        self.lr = lr
        self.base_lr = base_lr
        self.base_step_size = base_step_size
        self.base_config_calls = 0
        self.configure_calls = 0
        self.excluded_params: set[int] = set()

    @override
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]:
        self.base_config_calls += 1
        return (
            OptimizerConfig(
                name="SGD",
                params={
                    "lr": self.base_lr,
                    "momentum": 0.25,
                    "nesterov": False,
                },
            ),
            SchedulerConfig(
                name="StepLR",
                params={"step_size": self.base_step_size, "gamma": 0.6},
            ),
        )

    @override
    def configure_optimizers(
        self, excluded_params: set[int] | None = None
    ) -> tuple[
        Sequence[Optimizer],
        Sequence[LRSchedulerTypeUnion | LRSchedulerConfig],
    ]:
        self.configure_calls += 1
        self.excluded_params = set(excluded_params or set())
        params = [
            param
            for param in self.pl_module.parameters()
            if param.requires_grad and id(param) not in self.excluded_params
        ]
        if not params:
            return [], []
        optimizer = AdamW(params, lr=self.lr)
        return [optimizer], [ConstantLR(optimizer, factor=1.0)]

    @override
    def update_parameters(self) -> None:
        return None


def _capturing_strategy(
    snapshot: OptimizerSnapshot,
) -> CapturingFinetuningStrategy:
    strategy = snapshot.model.lightning_module.training_strategy
    assert isinstance(strategy, CapturingFinetuningStrategy)
    return cast(CapturingFinetuningStrategy, strategy)


def _build_model(cfg: Params, opts: Params) -> lxt.LuxonisModel:
    return lxt.LuxonisModel(
        cfg,
        opts | {"loader.params.n_classes": 10},
        allow_empty_dataset=True,
    )


def _has_gradient_accumulation_callback(model: lxt.LuxonisModel) -> bool:
    return any(
        isinstance(callback, GradientAccumulationScheduler)
        for callback in model.lightning_module.configure_callbacks()
    )


def test_configure_callbacks_does_not_build_strategy_optimizers(opts: Params):
    """Callback wiring must ask the strategy only for its
    *base configuration* (used to decide optimizer counts for
    accumulation), not to actually build the optimizers — building
    them here would double-instantiate optimizers on every
    ``configure_callbacks`` call, wasting memory and losing state.

    Setup:
        Head Linear rule + ``CapturingFinetuningStrategy`` (counts
        each of its own method invocations).

    Expected result:
        After model construction: ``get_base_configs`` was called once
        (for callback wiring), ``configure_optimizers`` was not called
        yet. After the explicit ``configure_optimizers`` call: both
        counters are exactly 1. Also, the accumulation callback is
        omitted because the finetuning rule forces >1 optimizer.
    """
    model = _build_model(
        config(
            [tiny_head_node({"parameters": [{"module_type": "Linear"}]})],
            trainer={
                "accumulate_grad_batches": 2,
                "training_strategy": {
                    "name": "CapturingFinetuningStrategy",
                    "params": {"lr": 0.07},
                },
            },
        ),
        opts,
    )
    strategy = model.lightning_module.training_strategy
    assert isinstance(strategy, CapturingFinetuningStrategy)

    assert not _has_gradient_accumulation_callback(model)
    assert strategy.base_config_calls == 1
    assert strategy.configure_calls == 0

    model.lightning_module.configure_optimizers()

    assert strategy.base_config_calls == 1
    assert strategy.configure_calls == 1


def test_gradient_accumulation_callback_uses_optimizer_count(opts: Params):
    """``GradientAccumulationScheduler`` only makes sense with a single
    optimizer — Lightning does not support it under manual optimization
    with multiple optimizers.

    Setup:
        Both models set ``accumulate_grad_batches=2`` on the trainer.
        - ``single_optimizer``: rule with no optimizer override → all
          rules share one Adam+ConstantLR key → one optimizer overall
          (default rule collapses into it).
        - ``multiple_optimizers``: rule overrides optimizer to SGD →
          two distinct optimizers (SGD for Linear, Adam default for
          the rest).

    Expected result:
        The callback is present only in the single-optimizer model.
        Both models still report ``automatic_optimization=True``
        immediately after construction — the switch to manual
        optimization happens later, inside ``configure_optimizers``,
        which this test does not invoke.
    """
    single_optimizer = _build_model(
        config(
            [tiny_head_node({"parameters": [{"module_type": "Linear"}]})],
            trainer={"accumulate_grad_batches": 2},
        ),
        opts,
    )
    multiple_optimizers = _build_model(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"name": "SGD"},
                    }
                )
            ],
            trainer={"accumulate_grad_batches": 2},
        ),
        opts,
    )

    assert _has_gradient_accumulation_callback(single_optimizer)
    assert not _has_gradient_accumulation_callback(multiple_optimizers)
    assert single_optimizer.lightning_module.automatic_optimization is True
    assert multiple_optimizers.lightning_module.automatic_optimization is True


def test_triple_lr_optimizer_count_omits_strategy_when_finetuning_claims_all(
    opts: Params,
):
    """When a finetuning rule claims *every* trainable parameter, the
    strategy has nothing left to optimize and drops out — that collapses
    the total optimizer count back to 1, which re-enables the
    ``GradientAccumulationScheduler`` callback.

    Setup:
        Head rule with no ``parameters`` (matches everything) using
        AdamW, plus TripleLRSGDStrategy and
        ``accumulate_grad_batches=2``.

    Expected result:
        The accumulation callback is installed because only one
        optimizer (the AdamW from the rule) is present — the
        strategy's SGD would have been the second, but it received no
        parameters and was skipped.
    """
    model = _build_model(
        config(
            [
                tiny_head_node(
                    {
                        "optimizer": {
                            "name": "AdamW",
                            "params": {"lr": 0.005},
                        }
                    }
                )
            ],
            trainer={
                "accumulate_grad_batches": 2,
                "training_strategy": {
                    "name": "TripleLRSGDStrategy",
                    "params": {"lr": 0.02},
                },
            },
        ),
        opts,
    )

    assert _has_gradient_accumulation_callback(model)


def test_manual_multi_optimizer_training_step_clips_gradients(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """Under manual optimization (multi-optimizer path) Lightning
    doesn't clip gradients for us — the training step must clip the
    whole model *once*, exactly like automatic optimization does.

    Clipping each optimizer separately would let the total applied
    gradient scale with the number of optimizers, so adding a single
    ``finetuning`` entry could make a previously converging run diverge.

    Setup:
        Head rule targeting Linear with SGD (forces two optimizers →
        manual mode) and ``gradient_clip_val=1.5``,
        ``gradient_clip_algorithm='value'``.

    Expected result:
        ``automatic_optimization`` flipped to ``False`` and the clipping
        happened in a single call covering every parameter that has a
        gradient, rather than once per optimizer.
    """
    model = _build_model(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"name": "SGD"},
                    }
                )
            ],
            trainer={
                "gradient_clip_val": 1.5,
                "gradient_clip_algorithm": "value",
            },
        ),
        opts,
    )
    module = model.lightning_module
    optimizers, _ = module.configure_optimizers()
    calls: list[tuple[set[int], float]] = []
    optimizer_accesses: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        module,
        "full_forward",
        lambda *_args, **_kwargs: SimpleNamespace(
            losses={"Head": {"CrossEntropyLoss": torch.tensor(1.0)}}
        ),
    )
    monkeypatch.setattr(
        luxonis_lightning,
        "compute_losses",
        lambda *_args, **_kwargs: (
            torch.tensor(1.0, requires_grad=True),
            {"loss": torch.tensor(1.0)},
        ),
    )

    def get_optimizers(*args: object, **kwargs: object) -> list[Optimizer]:
        optimizer_accesses.append((args, kwargs))
        return list(optimizers)

    monkeypatch.setattr(module, "optimizers", get_optimizers)
    monkeypatch.setattr(module, "lr_schedulers", list)
    monkeypatch.setattr(module, "manual_backward", lambda _loss: None)
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_value_",
        lambda parameters, clip_value: calls.append(
            ({id(parameter) for parameter in parameters}, clip_value)
        ),
    )
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    # Captured up front: `optimizer.zero_grad()` clears the gradients
    # again once the step has run.
    expected_ids = {id(parameter) for parameter in module.parameters()}
    module._trainer = SimpleNamespace(is_last_batch=False)  # type: ignore[assignment]

    module.training_step((torch.empty(0), {}), batch_idx=0)

    assert module.automatic_optimization is False
    assert optimizer_accesses == [((), {})]
    assert len(optimizers) > 1
    assert calls == [(expected_ids, 1.5)]


def test_manual_training_step_accepts_single_optimizer_and_scheduler(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """``training_step`` must tolerate ``optimizers``/``lr_schedulers``
    returning a single object (not a list) when there's exactly one of
    each — Lightning returns them that way in single-optimizer mode.

    Setup:
        Head with no finetuning → one optimizer, one scheduler.
        Manual optimization is forced on so the scheduler stepping
        code path actually runs on the last batch.

    Expected result:
        The scheduler's ``last_epoch`` advances by 1 — proving the
        single-scheduler branch reached ``scheduler.step()`` without
        tripping on the non-list return type.
    """
    model = _build_model(config([tiny_head_node()]), opts)
    module = model.lightning_module
    optimizers, _ = module.configure_optimizers()
    optimizer = next(iter(optimizers))
    scheduler = ConstantLR(optimizer, factor=1.0)
    initial_epoch = scheduler.last_epoch

    monkeypatch.setattr(
        module,
        "full_forward",
        lambda *_args, **_kwargs: SimpleNamespace(
            losses={"Head": {"CrossEntropyLoss": torch.tensor(1.0)}}
        ),
    )
    monkeypatch.setattr(
        luxonis_lightning,
        "compute_losses",
        lambda *_args, **_kwargs: (
            torch.tensor(1.0, requires_grad=True),
            {"loss": torch.tensor(1.0)},
        ),
    )
    monkeypatch.setattr(
        module, "optimizers", lambda *_args, **_kwargs: optimizer
    )
    monkeypatch.setattr(module, "lr_schedulers", lambda: scheduler)
    monkeypatch.setattr(module, "manual_backward", lambda _loss: None)
    module._trainer = SimpleNamespace(is_last_batch=True)  # type: ignore[assignment]
    module.automatic_optimization = False

    module.training_step((torch.empty(0), {}))

    assert scheduler.last_epoch == initial_epoch + 1


def test_strategy_base_configs_are_inherited_by_finetuning_rules(opts: Params):
    """When a training strategy is active, its ``get_base_configs()``
    supplies the *base* optimizer/scheduler seen by all finetuning rules
    — overriding whatever the user set in ``trainer.optimizer`` /
    ``trainer.scheduler``.

    Setup:
        - Trainer sets Adam(lr=0.9) + ConstantLR(factor=0.5) — these
          should end up unused as bases.
        - ``CapturingFinetuningStrategy`` returns SGD(lr=0.031,
          momentum=0.25) + StepLR(step_size=11, gamma=0.6) as its base
          configs, and its own ``configure_optimizers`` builds an
          AdamW for whatever's left.
        - The Head Linear rule inherits from those strategy bases
          without overriding anything.

    Expected result:
        Two optimizers: a first SGD from the finetuning rule
        (inheriting the strategy base's lr=0.031 and momentum=0.25)
        with a StepLR(step_size=11, gamma=0.6), and a second AdamW
        from the strategy carrying every remaining trainable
        parameter. The strategy is called exactly once for base
        configs and once to build its optimizer, and it receives the
        Linear.fc param ids as ``excluded_params`` so they don't
        appear in both optimizers.
    """
    snapshot = build_snapshot(
        config(
            [tiny_head_node({"parameters": [{"module_type": "Linear"}]})],
            trainer={
                "optimizer": {"name": "AdamW", "params": {"lr": 0.9}},
                "scheduler": {
                    "name": "ConstantLR",
                    "params": {"factor": 0.5},
                },
                "training_strategy": {
                    "name": "CapturingFinetuningStrategy",
                    "params": {
                        "lr": 0.07,
                        "base_lr": 0.031,
                        "base_step_size": 11,
                    },
                },
            },
        ),
        opts,
    )

    strategy = _capturing_strategy(snapshot)
    fc_names = matching_names(snapshot, "Head.Linear.fc")
    idx, optimizer, group = find_group(snapshot, fc_names)
    finetuning_scheduler = scheduler(snapshot.schedulers[idx])

    assert idx == 0
    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 2
    assert [type(opt) for opt in snapshot.optimizers] == [SGD, AdamW]
    assert [
        type(scheduler_cfg)
        for scheduler_cfg in map(scheduler, snapshot.schedulers)
    ] == [StepLR, ConstantLR]
    assert isinstance(optimizer, SGD)
    assert_group_options(group, {"lr": 0.031, "momentum": 0.25})
    assert finetuning_scheduler.step_size == 11
    assert finetuning_scheduler.gamma == pytest.approx(0.6)
    assert strategy.base_config_calls == 1
    assert strategy.configure_calls == 1
    assert names_for_ids(snapshot, strategy.excluded_params) == fc_names
    assert optimizer_names(snapshot, snapshot.optimizers[1]) == (
        trainable_parameter_names(snapshot) - fc_names
    )
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_strategy_receives_exact_ids_claimed_by_overlapping_finetuning_rules(
    opts: Params,
):
    """The bookkeeping between overlapping finetuning rules and the
    strategy must agree: the strategy sees exactly the set of
    parameter ids the finetuning pass claimed — no more, no less.

    Setup:
        - Head has two rules that would overlap:
          ``name='branch1'`` (lr=0.001) then ``module_type='Conv2d'``
          (lr=0.002). Rule 1 claims branch1 first; rule 2 picks up
          the remaining Conv2d.
        - ``CapturingFinetuningStrategy`` handles what's left (Linear).

    Expected result:
        - Two optimizers: a first SGD (from the strategy base config)
          holding both Conv2d groups, and a second AdamW (from the
          strategy) holding just the Linear params.
        - Strategy's ``excluded_params`` equals the union of both
          Conv2d rules' claims (i.e. every Conv2d id in the Head, not
          just branch1) — confirming that overlapping rules feed a
          single accurate "already claimed" set to the strategy
          rather than leaking the shadowed rule-2 matches.
    """
    snapshot = build_snapshot(
        config(
            [
                tiny_head_node(
                    [
                        {
                            "parameters": [{"name": "branch1"}],
                            "optimizer": {"params": {"lr": 0.001}},
                        },
                        {
                            "parameters": [{"module_type": "Conv2d"}],
                            "optimizer": {"params": {"lr": 0.002}},
                        },
                    ]
                )
            ],
            trainer={
                "training_strategy": {
                    "name": "CapturingFinetuningStrategy",
                    "params": {"lr": 0.07},
                }
            },
        ),
        opts,
    )

    strategy = _capturing_strategy(snapshot)
    branch1 = matching_names(snapshot, "Head.Conv2d.branch1")
    all_convs = matching_names(snapshot, "Head.Conv2d")
    remaining_convs = all_convs - branch1
    remaining_for_strategy = trainable_parameter_names(snapshot) - all_convs

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 2
    assert [type(opt) for opt in snapshot.optimizers] == [SGD, AdamW]
    assert [
        type(scheduler_cfg)
        for scheduler_cfg in map(scheduler, snapshot.schedulers)
    ] == [StepLR, ConstantLR]
    assert optimizer_group_names(snapshot, snapshot.optimizers[0]) == [
        branch1,
        remaining_convs,
    ]
    assert names_for_ids(snapshot, strategy.excluded_params) == all_convs
    assert strategy.excluded_params == ids_for_names(snapshot, all_convs)
    assert optimizer_names(snapshot, snapshot.optimizers[1]) == (
        remaining_for_strategy
    )
    assert optimizer_names(snapshot, snapshot.optimizers[1]).isdisjoint(
        all_convs
    )

    _, _, branch1_group = find_group(snapshot, branch1)
    _, _, remaining_conv_group = find_group(snapshot, remaining_convs)
    assert_group_options(branch1_group, {"lr": 0.001})
    assert_group_options(remaining_conv_group, {"lr": 0.002})
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_triple_lr_strategy_optimizer_contains_only_remaining_trainable_params(
    opts: Params,
):
    """Real-world strategy variant of the "strategy + finetuning"
    interaction using the production ``TripleLRSGDStrategy``.

    Setup:
        - Head Linear rule (no optimizer override, so it uses the
          strategy's SGD base).
        - ``TripleLRSGDStrategy`` builds an SGD with three
          conventional groups (bn, weight, bias) plus LambdaLR
          scheduling.

    Expected result:
        Two SGD optimizers with LambdaLR schedulers:
        1. The finetuning-built SGD holding the Linear.fc params.
        2. The strategy-built SGD with its three-group layout
           (empty bn group in this tiny model, weights, biases).
        The strategy's optimizer holds *only* the params not already
        claimed by finetuning — i.e. everything but Linear.fc.
    """
    snapshot = build_snapshot(
        config(
            [tiny_head_node({"parameters": [{"module_type": "Linear"}]})],
            trainer={
                "training_strategy": {
                    "name": "TripleLRSGDStrategy",
                    "params": {"lr": 0.02},
                }
            },
        ),
        opts,
    )

    fc_names = matching_names(snapshot, "Head.Linear.fc")
    strategy_names = trainable_parameter_names(snapshot) - fc_names

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 2
    assert [type(opt) for opt in snapshot.optimizers] == [SGD, SGD]
    assert all(
        isinstance(scheduler_cfg, LambdaLR)
        for scheduler_cfg in map(scheduler, snapshot.schedulers)
    )
    assert optimizer_names(snapshot, snapshot.optimizers[0]) == fc_names
    assert optimizer_names(snapshot, snapshot.optimizers[1]) == strategy_names
    assert optimizer_group_names(snapshot, snapshot.optimizers[1]) == [
        set(),
        {name for name in strategy_names if name.endswith(".weight")},
        {name for name in strategy_names if name.endswith(".bias")},
    ]
    assert optimizer_names(snapshot, snapshot.optimizers[1]).isdisjoint(
        fc_names
    )
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_triple_lr_strategy_optimizer_is_omitted_when_finetuning_claims_all(
    opts: Params,
):
    """Mirror of the callback test above, verified at the optimizer
    level. When finetuning consumes every trainable parameter, the
    strategy's ``configure_optimizers`` receives an empty set and must
    return no optimizer at all — the finetuning optimizer is the only
    one built.

    Setup:
        Head rule with no ``parameters`` selector (matches everything)
        overriding to AdamW, plus TripleLRSGDStrategy.

    Expected result:
        One optimizer, an AdamW from the finetuning rule, with the
        strategy's LambdaLR scheduling applied. The strategy's SGD
        drops out because it has zero parameters to optimize.
    """
    snapshot = build_snapshot(
        config(
            [
                tiny_head_node(
                    {
                        "optimizer": {
                            "name": "AdamW",
                            "params": {"lr": 0.005},
                        }
                    }
                )
            ],
            trainer={
                "training_strategy": {
                    "name": "TripleLRSGDStrategy",
                    "params": {"lr": 0.02},
                }
            },
        ),
        opts,
    )

    all_names = trainable_parameter_names(snapshot)

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 1
    assert isinstance(snapshot.optimizers[0], AdamW)
    assert isinstance(scheduler(snapshot.schedulers[0]), LambdaLR)
    assert optimizer_names(snapshot, snapshot.optimizers[0]) == all_names
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_reduce_on_plateau_min_mode_steps_with_validation_loss(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """Under manual optimization Lightning does not auto-step
    ReduceLROnPlateau, so `_step_reduce_lr_on_plateau_schedulers` drives
    it from the validation epoch end.

    In ``min`` mode the scheduler tracks the validation loss, so the
    helper must feed the loss directly to ``scheduler.step`` — bypassing
    the main-metric lookup entirely (a bug there would still consume
    plateau ticks with the wrong signal).
    """
    model = _build_model(config([tiny_head_node()]), opts)
    module = model.lightning_module
    module.automatic_optimization = False

    optim = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    plateau = ReduceLROnPlateau(optim, mode="min")
    step_calls: list[float] = []
    monkeypatch.setattr(plateau, "step", step_calls.append)
    monkeypatch.setattr(module, "lr_schedulers", lambda: plateau)

    module._step_reduce_lr_on_plateau_schedulers(
        loss=0.42, metrics={"Head": {"Accuracy": 0.9}}
    )

    assert step_calls == [0.42]


def test_reduce_on_plateau_max_mode_steps_with_main_metric_value(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """In ``max`` mode ReduceLROnPlateau watches a monotonically-
    increasing metric.

    The helper must resolve ``nodes.main_metric`` into the correct
    scalar from the logged metrics table and pass *that* (not the loss)
    to ``scheduler.step``. Getting this wrong would cause plateau
    detection to fire on the wrong signal.
    """
    model = _build_model(config([tiny_head_node()]), opts)
    module = model.lightning_module
    module.automatic_optimization = False
    module.nodes.main_metric = MainMetric("Head", "Accuracy")

    optim = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    plateau = ReduceLROnPlateau(optim, mode="max")
    step_calls: list[float] = []
    monkeypatch.setattr(plateau, "step", step_calls.append)
    monkeypatch.setattr(module, "lr_schedulers", lambda: plateau)

    module._step_reduce_lr_on_plateau_schedulers(
        loss=0.42, metrics={"Head": {"Accuracy": 0.87}}
    )

    assert step_calls == [0.87]


def test_reduce_on_plateau_max_mode_without_main_metric_raises(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """``max`` mode is nonsensical without a metric to watch.

    Silently falling back to the loss would mask a misconfiguration, so
    the helper surfaces the mismatch as a ``RuntimeError`` at the first
    validation epoch end.
    """
    model = _build_model(config([tiny_head_node()]), opts)
    module = model.lightning_module
    module.automatic_optimization = False
    module.nodes.main_metric = None

    optim = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    plateau = ReduceLROnPlateau(optim, mode="max")
    monkeypatch.setattr(module, "lr_schedulers", lambda: plateau)

    with pytest.raises(RuntimeError, match="without a main metric"):
        module._step_reduce_lr_on_plateau_schedulers(
            loss=0.1, metrics={"Head": {"Accuracy": 0.9}}
        )


def test_reduce_on_plateau_max_mode_missing_metric_value_raises(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """Main metrics that don't reduce to a plain scalar reachable via
    ``metrics[node][name]`` (multi-value returns, custom aggregation, or
    metrics that weren't logged) manifest here as a ``KeyError``.

    The helper converts this into a ``ValueError`` with a message
    pointing at the underlying misconfiguration instead of a bare
    ``KeyError`` escaping from Lightning's validation epoch end.
    """
    model = _build_model(config([tiny_head_node()]), opts)
    module = model.lightning_module
    module.automatic_optimization = False
    module.nodes.main_metric = MainMetric("Head", "Accuracy")

    optim = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    plateau = ReduceLROnPlateau(optim, mode="max")
    monkeypatch.setattr(module, "lr_schedulers", lambda: plateau)

    with pytest.raises(ValueError, match="not a logged scalar"):
        module._step_reduce_lr_on_plateau_schedulers(
            loss=0.1, metrics={"Head": {}}
        )


def test_reduce_on_plateau_step_is_noop_under_automatic_optimization(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """Under automatic optimization Lightning already steps schedulers
    itself.

    Running our manual helper would step the scheduler twice per
    validation epoch — the early-return guard prevents that. Regressions
    here are silent (LR decays too fast) so a direct assertion is worth
    the cost.
    """
    model = _build_model(config([tiny_head_node()]), opts)
    module = model.lightning_module
    module.automatic_optimization = True

    optim = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    plateau = ReduceLROnPlateau(optim, mode="min")
    step_calls: list[float] = []
    monkeypatch.setattr(plateau, "step", step_calls.append)
    monkeypatch.setattr(module, "lr_schedulers", lambda: plateau)

    module._step_reduce_lr_on_plateau_schedulers(
        loss=0.42, metrics={"Head": {"Accuracy": 0.9}}
    )

    assert step_calls == []


def test_manual_training_step_skips_scheduler_step_when_not_last_batch(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """Non-plateau schedulers should step exactly once per epoch — at
    the last training batch, matching Lightning's automatic behaviour.

    Any other batch during the epoch must leave them untouched;
    otherwise LRs would decay per-step rather than per-epoch and
    completely undo the configured schedule.
    """
    model = _build_model(
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
    optimizers, schedulers = module.configure_optimizers()
    scheduler_objects = [scheduler(s) for s in schedulers]
    step_calls: list[int] = []
    for i, sched in enumerate(scheduler_objects):
        monkeypatch.setattr(sched, "step", lambda i=i: step_calls.append(i))

    monkeypatch.setattr(
        module,
        "full_forward",
        lambda *_args, **_kwargs: SimpleNamespace(
            losses={"Head": {"CrossEntropyLoss": torch.tensor(1.0)}}
        ),
    )
    monkeypatch.setattr(
        luxonis_lightning,
        "compute_losses",
        lambda *_args, **_kwargs: (
            torch.tensor(1.0, requires_grad=True),
            {"loss": torch.tensor(1.0)},
        ),
    )
    monkeypatch.setattr(
        module, "optimizers", lambda *_args, **_kwargs: list(optimizers)
    )
    monkeypatch.setattr(
        module, "lr_schedulers", lambda: list(scheduler_objects)
    )
    monkeypatch.setattr(module, "manual_backward", lambda _loss: None)
    module._trainer = SimpleNamespace(is_last_batch=False)  # type: ignore[assignment]

    module.training_step((torch.empty(0), {}))

    assert step_calls == []


def test_manual_training_step_skips_clipping_when_gradient_clip_val_none(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """When the user has not set ``gradient_clip_val`` the manual path
    must not call ``clip_gradients`` at all.

    ``clip_gradients`` has non-trivial cost (an extra pass over every
    parameter) and, for exotic optimizers, may error on parameters it
    doesn't understand — so the ``is not None`` guard is load-bearing.
    """
    model = _build_model(
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
    clip_calls: list[object] = []

    assert module.cfg.trainer.gradient_clip_val is None

    monkeypatch.setattr(
        module,
        "full_forward",
        lambda *_args, **_kwargs: SimpleNamespace(
            losses={"Head": {"CrossEntropyLoss": torch.tensor(1.0)}}
        ),
    )
    monkeypatch.setattr(
        luxonis_lightning,
        "compute_losses",
        lambda *_args, **_kwargs: (
            torch.tensor(1.0, requires_grad=True),
            {"loss": torch.tensor(1.0)},
        ),
    )
    monkeypatch.setattr(
        module, "optimizers", lambda *_args, **_kwargs: list(optimizers)
    )
    monkeypatch.setattr(module, "lr_schedulers", list)
    monkeypatch.setattr(module, "manual_backward", lambda _loss: None)
    monkeypatch.setattr(
        module,
        "clip_gradients",
        lambda *args, **kwargs: clip_calls.append(args),
    )
    module._trainer = SimpleNamespace(is_last_batch=False)  # type: ignore[assignment]

    module.training_step((torch.empty(0), {}))

    assert clip_calls == []


def test_manual_training_step_does_not_step_reduce_lr_on_plateau_scheduler(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
    """``ReduceLROnPlateau`` is driven exclusively from the validation
    epoch end via ``_step_reduce_lr_on_plateau_schedulers`` — the
    training step's ``is_last_batch`` branch must skip it.

    Stepping it here would consume a "no-improvement" tick per training
    epoch before the validation metrics are even computed, which drives
    the LR down long before the plateau it's meant to detect actually
    occurs.
    """
    model = _build_model(config([tiny_head_node()]), opts)
    module = model.lightning_module
    optimizers, _ = module.configure_optimizers()

    plateau = ReduceLROnPlateau(optimizers[0], mode="min")
    step_calls: list[object] = []
    monkeypatch.setattr(
        plateau, "step", lambda *args, **kwargs: step_calls.append(args)
    )

    monkeypatch.setattr(
        module,
        "full_forward",
        lambda *_args, **_kwargs: SimpleNamespace(
            losses={"Head": {"CrossEntropyLoss": torch.tensor(1.0)}}
        ),
    )
    monkeypatch.setattr(
        luxonis_lightning,
        "compute_losses",
        lambda *_args, **_kwargs: (
            torch.tensor(1.0, requires_grad=True),
            {"loss": torch.tensor(1.0)},
        ),
    )
    monkeypatch.setattr(
        module, "optimizers", lambda *_args, **_kwargs: list(optimizers)
    )
    monkeypatch.setattr(module, "lr_schedulers", lambda: plateau)
    monkeypatch.setattr(module, "manual_backward", lambda _loss: None)
    module._trainer = SimpleNamespace(is_last_batch=True)  # type: ignore[assignment]
    module.automatic_optimization = False

    module.training_step((torch.empty(0), {}))

    assert step_calls == []
