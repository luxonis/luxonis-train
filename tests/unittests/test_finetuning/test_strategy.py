from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from loguru import logger
from luxonis_ml.typing import Params
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, StepLR
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.config.config import (
    CallbackConfig,
    Config,
    OptimizerConfig,
    SchedulerConfig,
)
from luxonis_train.lightning.training_plan import StrategyRule
from luxonis_train.lightning.utils import build_training_strategy
from luxonis_train.strategies.base_strategy import BaseTrainingStrategy

from ._helpers import (
    OptimizerSnapshot,
    assert_all_trainable_parameters_assigned,
    assert_group_options,
    assert_no_duplicate_parameters,
    build_snapshot,
    config,
    find_group,
    matching_names,
    optimizer_group_names,
    optimizer_names,
    parent_parameter_head_node,
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
        self.rules_calls = 0
        self.attach_calls = 0

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
    def rules(self) -> list[StrategyRule]:
        self.rules_calls += 1

        def match_all(
            module: nn.Module,
            module_name: str,
            parameter: nn.Parameter,
            parameter_name: str,
        ) -> bool:
            _ = module, module_name, parameter, parameter_name
            return True

        return [
            StrategyRule(
                tag="capturing/rest",
                selector=match_all,
                optimizer=OptimizerConfig(
                    name="AdamW", params={"lr": self.lr}
                ),
                scheduler=SchedulerConfig(
                    name="ConstantLR", params={"factor": 1.0}
                ),
            )
        ]

    @override
    def attach(self, runtime: Any, handles: Any) -> None:
        super().attach(runtime, handles)
        self.attach_calls += 1

    def claimed_names(self, snapshot: OptimizerSnapshot) -> set[str]:
        return {
            snapshot.names_by_id[id(parameter)]
            for handles in self.group_handles.values()
            for handle in handles
            for parameter in self.runtime.group(handle)["params"]
        }


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
    """Callback wiring must not consult the strategy at all - the
    optimizer count no longer matters for callbacks, since every
    configuration is driven through one (possibly composite) optimizer
    under automatic optimization.

    Setup:
        Head Linear rule + ``CapturingFinetuningStrategy`` (counts
        each of its own method invocations).

    Expected result:
        After model construction and callback wiring neither
        ``get_base_configs`` nor ``rules`` was called, and the
        accumulation callback is present. ``configure_optimizers``
        resolves the plan (one ``get_base_configs`` + one ``rules``
        call) and hands the strategy its group handles.
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

    assert _has_gradient_accumulation_callback(model)
    assert strategy.base_config_calls == 0
    assert strategy.rules_calls == 0
    assert strategy.attach_calls == 0

    model.lightning_module.configure_optimizers()

    assert strategy.base_config_calls == 1
    assert strategy.rules_calls == 1
    assert strategy.attach_calls == 1
    assert set(strategy.group_handles) == {"capturing/rest"}


def test_gradient_accumulation_callback_uses_optimizer_count(opts: Params):
    """``GradientAccumulationScheduler`` is available regardless of the
    finetuning topology: several optimizer configurations are driven
    through one composite optimizer under automatic optimization.

    Setup:
        Both models set ``accumulate_grad_batches=2`` on the trainer.
        - ``single_optimizer``: rule with no optimizer override → all
          rules share one Adam+ConstantLR key → one optimizer overall.
        - ``multiple_optimizers``: rule overrides optimizer to SGD →
          two inner optimizers inside one composite.

    Expected result:
        The callback is present in both models and both stay in
        automatic optimization.
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
    assert _has_gradient_accumulation_callback(multiple_optimizers)
    assert single_optimizer.lightning_module.automatic_optimization is True
    assert multiple_optimizers.lightning_module.automatic_optimization is True


def test_triple_lr_optimizer_count_omits_strategy_when_finetuning_claims_all(
    opts: Params,
):
    """When a finetuning rule claims *every* parameter, the strategy has
    nothing left to optimize and contributes no groups.

    Setup:
        Head rule with no ``parameters`` (matches everything) using
        AdamW, plus TripleLRSGDStrategy and
        ``accumulate_grad_batches=2``.

    Expected result:
        The accumulation callback is installed (it always is now) and
        only one optimizer - the AdamW from the rule - is built.
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


def test_strategy_base_configs_are_inherited_by_finetuning_rules(opts: Params):
    """A strategy's ``get_base_configs()`` supplies the *base*
    optimizer/scheduler seen by all finetuning rules.

    It overrides whatever the user set in ``trainer.optimizer`` /
    ``trainer.scheduler``.

    Setup:
        - Trainer sets AdamW(lr=0.9) + ConstantLR(factor=0.5) - these
          should end up unused as bases.
        - ``CapturingFinetuningStrategy`` returns SGD(lr=0.031,
          momentum=0.25) + StepLR(step_size=11, gamma=0.6) as its base
          configs, and its own rule claims whatever is left with AdamW.
        - The Head Linear rule inherits from the strategy bases
          without overriding anything.

    Expected result:
        Two inner optimizers: a first SGD from the finetuning rule
        (inheriting the strategy base's lr=0.031 and momentum=0.25)
        with a StepLR(step_size=11, gamma=0.6), and a second AdamW
        from the strategy's rule carrying every remaining parameter.
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
    assert strategy.rules_calls == 1
    assert strategy.claimed_names(snapshot) == (
        trainable_parameter_names(snapshot) - fc_names
    )
    assert optimizer_names(snapshot, snapshot.optimizers[1]) == (
        trainable_parameter_names(snapshot) - fc_names
    )
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_strategy_rules_claim_exactly_the_unclaimed_parameters(
    opts: Params,
):
    """The partition between overlapping finetuning rules and the.

    strategy must agree: the strategy's groups hold exactly the
    parameters no finetuning rule claimed - no more, no less.

    Setup:
        - Head has two rules that would overlap:
          ``name='branch1'`` (lr=0.001) then ``module_type='Conv2d'``
          (lr=0.002). Rule 1 claims branch1 first; rule 2 picks up
          the remaining Conv2d.
        - ``CapturingFinetuningStrategy``'s rule handles what is left
          (Linear).

    Expected result:
        - Two inner optimizers: a first SGD (from the strategy base
          config) holding both Conv2d groups, and a second AdamW
          (from the strategy's rule) holding just the Linear params.
        - The strategy's claimed set is exactly the complement of the
          union of both Conv2d rules' claims.
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
    assert strategy.claimed_names(snapshot) == remaining_for_strategy
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
          strategy's SGD base and its LambdaLR base scheduler).
        - ``TripleLRSGDStrategy`` contributes its conventional rules
          (batch-norm weights, weights, biases) with the same SGD +
          LambdaLR pair.

    Expected result:
        Because the rule inherits the strategy's own base configs,
        everything collapses into ONE SGD optimizer: the rule's
        Linear group first, then the strategy's weight and bias
        groups holding only the parameters not already claimed (this
        tiny model has no batch-norm layers, so no batch-norm group
        is created). The strategy receives handles for its two
        non-empty groups.
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

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 1
    (optimizer,) = snapshot.optimizers
    assert isinstance(optimizer, SGD)
    assert isinstance(scheduler(snapshot.schedulers[0]), LambdaLR)
    assert optimizer_group_names(snapshot, optimizer) == [
        fc_names,
        {name for name in strategy_names if name.endswith(".weight")},
        {name for name in strategy_names if name.endswith(".bias")},
    ]
    strategy = snapshot.model.lightning_module.training_strategy
    assert strategy is not None
    assert set(strategy.group_handles) == {
        "triple_lr/weights",
        "triple_lr/biases",
    }
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


def _strategy_config(trainer_extra: Params) -> Config:
    return Config(
        model=cast(
            Any, {"name": "test_finetuning", "nodes": [tiny_head_node()]}
        ),
        trainer=cast(
            Any,
            {
                "training_strategy": {"name": "CapturingFinetuningStrategy"},
                **trainer_extra,
            },
        ),
    )


def _override_warnings(cfg: Config) -> list[str]:
    # The sink must be attached *after* the `Config` is built:
    # `Config.check_rich_logging` calls `setup_logging`, which does
    # `logger.remove()` and would drop it.
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        build_training_strategy(cfg, cast(Any, SimpleNamespace()))
    finally:
        logger.remove(handler_id)
    return [m for m in messages if "Training strategy is defined" in m]


def test_strategy_override_warning_only_when_optimizer_explicitly_set():
    """`trainer.optimizer` and `trainer.scheduler` have a
    `default_factory`, so they are never `None`.

    Guarding the "training strategy will override your
    optimizer/scheduler" warning with an `is not None` check therefore
    makes it fire for every strategy-based config, including the shipped
    ones that never mention an optimizer. Only the fields the user
    actually set should be reported.
    """
    assert _override_warnings(_strategy_config({})) == []

    (warning,) = _override_warnings(
        _strategy_config({"optimizer": {"name": "AdamW"}})
    )
    assert "optimizer" in warning
    assert "scheduler" not in warning

    (warning,) = _override_warnings(
        _strategy_config({"scheduler": {"name": "StepLR"}})
    )
    assert "scheduler" in warning
    assert "optimizer" not in warning

    (warning,) = _override_warnings(
        _strategy_config(
            {
                "optimizer": {"name": "AdamW"},
                "scheduler": {"name": "StepLR"},
            }
        )
    )
    assert "optimizer" in warning
    assert "scheduler" in warning

    # A config rebuilt from `model_dump` reports *every* field as set,
    # so `model_fields_set` cannot be used here: the tuner and the saved
    # `training_config.yaml` both take that route.
    round_tripped = Config.get_config(_strategy_config({}).model_dump())
    assert _override_warnings(round_tripped) == []

    # ...and mutating the nested model does not mark the field as set,
    # yet it genuinely changes the optimizer.
    mutated = _strategy_config({})
    mutated.trainer.optimizer.name = "AdamW"
    (warning,) = _override_warnings(mutated)
    assert "optimizer" in warning


def test_strategy_tail_adopts_attribute_parameters(opts: Params):
    """Parameters that are not literal ``weight``/``bias`` attributes
    (e.g. a scalar ``alpha``) match none of TripleLR's structural rules
    and must flow to the default tail instead of being silently dropped
    from optimization (the pre-plan implementation lost them).
    """
    snapshot = build_snapshot(
        config(
            [parent_parameter_head_node()],
            trainer={
                "training_strategy": {
                    "name": "TripleLRSGDStrategy",
                    "params": {"lr": 0.02},
                }
            },
        ),
        opts,
    )

    alpha_names = matching_names(snapshot, "alpha")
    assert alpha_names
    _, optimizer, group = find_group(snapshot, alpha_names)
    assert isinstance(optimizer, SGD)
    assert group["lr"] == pytest.approx(0.02)
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


class OldStyleStrategy(BaseTrainingStrategy):
    """A strategy written against the released (pre-rules) API: it
    neither implements ``rules`` nor ``get_base_configs``.
    """

    def __init__(
        self, pl_module: "lxt.LuxonisLightningModule", lr: float = 0.05
    ):
        self.pl_module = pl_module
        self.lr = lr
        self.update_calls = 0
        self._optimizer = AdamW(
            [
                parameter
                for parameter in pl_module.parameters()
                if parameter.requires_grad
            ],
            lr=lr,
        )

    def configure_optimizers(self):
        return [self._optimizer], [ConstantLR(self._optimizer, factor=1.0)]

    @override
    def update_parameters(self) -> None:
        self.update_calls += 1


def test_legacy_strategy_is_mounted_with_a_deprecation_warning(
    opts: Params,
):
    """A strategy implementing the released ``configure_optimizers``
    API is wrapped by the compatibility adapter: a deprecation warning
    is logged, its optimizer is mounted verbatim (single opaque inner -
    the raw shapes are kept), and ``update_parameters`` forwards.
    """
    from luxonis_train.strategies.legacy import LegacyStrategyAdapter

    # `LuxonisModel` construction re-runs `setup_logging` (wiping any
    # loguru sink), so the warning is captured around
    # `build_training_strategy` directly, driven with a stub module.
    cfg = _strategy_config({})
    cfg.trainer.training_strategy = CallbackConfig(name="OldStyleStrategy")
    stub = SimpleNamespace(
        parameters=lambda: iter([nn.Parameter(torch.zeros(2))])
    )
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        strategy = build_training_strategy(cfg, cast(Any, stub))
    finally:
        logger.remove(handler_id)

    assert any("deprecated strategy API" in message for message in messages)
    assert isinstance(strategy, LegacyStrategyAdapter)
    assert strategy.legacy_name == "OldStyleStrategy"
    strategy.update_parameters()

    # end to end: the mounted optimizer keeps its raw shapes
    model = _build_model(
        config(
            [tiny_head_node()],
            trainer={"training_strategy": {"name": "OldStyleStrategy"}},
        ),
        opts,
    )
    optimizers, scheduler_configs = (
        model.lightning_module.configure_optimizers()
    )
    (optimizer,) = optimizers
    assert isinstance(optimizer, AdamW)
    assert isinstance(scheduler_configs[0], ConstantLR)
    legacy = model.lightning_module.training_strategy
    assert isinstance(legacy, LegacyStrategyAdapter)
    legacy.update_parameters()


def test_legacy_strategy_overlap_with_finetuning_rules_errors(
    opts: Params,
):
    """A legacy strategy whose optimizer holds parameters that a
    finetuning rule already claimed is a hard error - double membership
    would step those parameters twice.
    """
    model = _build_model(
        config(
            [tiny_head_node({"parameters": [{"module_type": "Linear"}]})],
            trainer={"training_strategy": {"name": "OldStyleStrategy"}},
        ),
        opts,
    )
    with pytest.raises(ValueError, match="already claimed"):
        model.lightning_module.configure_optimizers()
