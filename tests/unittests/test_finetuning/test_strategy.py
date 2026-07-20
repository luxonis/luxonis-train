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
    calls: list[tuple[Optimizer, float, str]] = []

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
        lambda optimizer, gradient_clip_val, gradient_clip_algorithm: (
            calls.append(
                (optimizer, gradient_clip_val, gradient_clip_algorithm)
            )
        ),
    )
    module._trainer = SimpleNamespace(is_last_batch=False)  # type: ignore[assignment]

    module.training_step((torch.empty(0), {}))

    assert module.automatic_optimization is False
    assert calls == [(optimizer, 1.5, "value") for optimizer in optimizers]


def test_manual_training_step_accepts_single_optimizer_and_scheduler(
    opts: Params, monkeypatch: pytest.MonkeyPatch
):
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
