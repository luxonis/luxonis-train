from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import Adam

from luxonis_train.config.config import Config, NodeConfig
from luxonis_train.lightning.optimization_planner import OptimizationPlanner


class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)


class DummyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)


class DummyNodeWrapper(nn.Module):
    def __init__(
        self,
        name: str,
        module: nn.Module,
        cfg: NodeConfig,
        task_name: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.module = module
        self.cfg = cfg
        self.task_name = task_name

    @property
    def formatted_name(self) -> str:
        if self.task_name:
            return f"{self.task_name}-{self.name}"
        return self.name


class DummyNodes(nn.ModuleDict):
    def __init__(
        self,
        cfg: Config,
        wrappers: dict[str, DummyNodeWrapper],
        main_metric: "DummyMainMetric | None" = None,
    ) -> None:
        super().__init__(wrappers)
        self.cfg = cfg
        self.main_metric = main_metric

    def formatted_name(self, node_name: str) -> str:
        return self[node_name].formatted_name


@dataclass(frozen=True)
class PlannedOptimization:
    optimizers: list[torch.optim.Optimizer]
    schedulers: list[Any]
    plan: Any


@dataclass(frozen=True)
class DummyMainMetric:
    node_name: str
    metric_name: str


def _cfg(config: dict[str, Any]) -> Config:
    return Config.get_config(
        config,
        {
            "rich_logging": False,
            "trainer.epochs": 3,
            "trainer.validation_interval": 1,
            "trainer.smart_cfg_auto_populate": False,
        },
    )


def _nodes(cfg: Config) -> DummyNodes:
    wrappers = {
        "Backbone": DummyNodeWrapper(
            "Backbone", DummyBackbone(), cfg.model.nodes[0]
        ),
        "Classifier": DummyNodeWrapper(
            "Classifier",
            DummyHead(),
            cfg.model.nodes[1],
            task_name="classification",
        ),
    }
    return DummyNodes(
        cfg, wrappers, main_metric=DummyMainMetric("Classifier", "Accuracy")
    )


def _build_planned_optimization(
    cfg: Config, nodes: DummyNodes
) -> PlannedOptimization:
    construction_attempts = (
        lambda: OptimizationPlanner(cfg, nodes),
        lambda: OptimizationPlanner(config=cfg, nodes=nodes),
        lambda: OptimizationPlanner(cfg=cfg, nodes=nodes),
        lambda: OptimizationPlanner(nodes=nodes, cfg=cfg),
    )
    last_error: Exception | None = None
    planner = None
    for construct in construction_attempts:
        try:
            planner = construct()
            break
        except TypeError as exc:
            last_error = exc
    if planner is None:
        raise AssertionError(
            "OptimizationPlanner construction API is unsupported by these "
            "tests."
        ) from last_error

    plan = None
    for method_name in ("build_plan", "build", "plan"):
        method = getattr(planner, method_name, None)
        if method is not None:
            plan = method()
            break
    if plan is None:
        raise AssertionError(
            "OptimizationPlanner must expose build_plan(), build(), or "
            "plan()."
        )

    optimizers = getattr(plan, "optimizers", None)
    schedulers = getattr(plan, "schedulers", None)
    if optimizers is None and isinstance(plan, tuple):
        optimizers = plan[0]
        schedulers = plan[1] if len(plan) > 1 else []

    assert optimizers is not None
    return PlannedOptimization(list(optimizers), list(schedulers or []), plan)


def _parameter_group_for(
    optimizer: torch.optim.Optimizer, parameter: nn.Parameter
) -> dict[str, Any]:
    parameter_id = id(parameter)
    groups = [
        group
        for group in optimizer.param_groups
        if any(
            id(group_parameter) == parameter_id
            for group_parameter in group["params"]
        )
    ]
    assert len(groups) == 1
    return groups[0]


def test_base_config_creates_one_optimizer_and_scheduler() -> None:
    cfg = _cfg(
        {
            "trainer": {
                "optimizer": {"name": "Adam", "params": {"lr": 0.001}},
                "scheduler": {
                    "name": "ConstantLR",
                    "params": {"factor": 1.0},
                },
            },
            "model": {
                "nodes": [
                    {"name": "Backbone"},
                    {
                        "name": "Classifier",
                        "metrics": [
                            {"name": "Accuracy", "is_main_metric": True}
                        ],
                    },
                ]
            },
        }
    )

    planned = _build_planned_optimization(cfg, _nodes(cfg))

    assert len(planned.optimizers) == 1
    assert isinstance(planned.optimizers[0], Adam)
    assert len(planned.schedulers) == 1


def test_finetuning_module_type_override_creates_parameter_group() -> None:
    cfg = _cfg(
        {
            "trainer": {
                "optimizer": {
                    "name": "Adam",
                    "params": {"lr": 0.001, "weight_decay": 0.0},
                },
                "scheduler": {
                    "name": "ConstantLR",
                    "params": {"factor": 1.0},
                },
            },
            "model": {
                "nodes": [
                    {"name": "Backbone"},
                    {
                        "name": "Classifier",
                        "metrics": [
                            {"name": "Accuracy", "is_main_metric": True}
                        ],
                        "finetuning": [
                            {
                                "parameters": [{"module_type": "Linear"}],
                                "optimizer": {
                                    "params": {
                                        "lr": 0.005,
                                        "weight_decay": 0.123,
                                    }
                                },
                            }
                        ],
                    },
                ]
            },
        }
    )
    nodes = _nodes(cfg)

    planned = _build_planned_optimization(cfg, nodes)

    assert len(planned.optimizers) == 1
    optimizer = planned.optimizers[0]
    linear_group = _parameter_group_for(
        optimizer, nodes["Classifier"].module.fc.weight
    )
    conv_group = _parameter_group_for(
        optimizer, nodes["Classifier"].module.conv.weight
    )
    assert linear_group["lr"] == pytest.approx(0.005)
    assert linear_group["weight_decay"] == pytest.approx(0.123)
    assert conv_group["lr"] == pytest.approx(0.001)
    assert conv_group["weight_decay"] == pytest.approx(0.0)


def test_reduce_lr_on_plateau_monitor_uses_formatted_node_name() -> None:
    cfg = _cfg(
        {
            "trainer": {
                "optimizer": {"name": "Adam", "params": {"lr": 0.001}},
                "scheduler": {
                    "name": "ReduceLROnPlateau",
                    "params": {"mode": "max"},
                },
            },
            "model": {
                "nodes": [
                    {"name": "Backbone"},
                    {
                        "name": "Classifier",
                        "metrics": [
                            {"name": "Accuracy", "is_main_metric": True}
                        ],
                    },
                ]
            },
        }
    )

    planned = _build_planned_optimization(cfg, _nodes(cfg))

    assert planned.schedulers
    scheduler_config = planned.schedulers[0]
    assert isinstance(scheduler_config, dict)
    assert (
        scheduler_config["monitor"]
        == "val/metric/classification-Classifier/Accuracy"
    )
