from collections.abc import Sequence
from typing import cast

import pytest
from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerTypeUnion,
)
from luxonis_ml.typing import Params
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, StepLR
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.config.config import OptimizerConfig, SchedulerConfig
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
