from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from lightning.pytorch.utilities.types import LRSchedulerConfig
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from loguru import logger
from torch import nn
from torch.optim import Optimizer

from luxonis_train.config import Config
from luxonis_train.config.config import (
    FinetuningConfig,
    OptimizerConfig,
    ParameterPattern,
    SchedulerConfig,
)

from .utils import MainMetric
from .utils import Nodes
from .utils import build_optimizer_scheduler
from .utils import merge_config_items


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    parameter: nn.Parameter
    node_name: str
    formatted_node_name: str
    module_name: str
    module_type: str
    parameter_name: str
    role: Literal["weight", "bias", "batch_norm_weight", "other"]
    requires_grad: bool
    frozen_until_epoch: int | None


@dataclass
class ParameterGroupSpec:
    params: list[nn.Parameter]
    optimizer_name: str
    optimizer_params: dict[str, Any]
    scheduler_name: str
    scheduler_params: dict[str, Any]
    source: str
    tags: set[str] = field(default_factory=set)
    strategy_role: str | None = None


@dataclass
class OptimizationPlan:
    optimizers: list[Optimizer]
    schedulers: list[LRSchedulerTypeUnion | LRSchedulerConfig]
    parameter_groups: list[ParameterGroupSpec]
    parameter_to_optimizer: dict[int, int]
    frozen_specs: list[ParameterSpec]
    uses_manual_optimization: bool
    strategy: Any | None


class OptimizationPlanner:
    def __init__(
        self,
        cfg: Config,
        nodes: Nodes,
        main_metric: MainMetric | None = None,
        training_strategy: Any | None = None,
    ):
        self.cfg = cfg
        self.nodes = nodes
        self.main_metric = main_metric or getattr(nodes, "main_metric", None)
        self.training_strategy = training_strategy
        self._plan: OptimizationPlan | None = None
        self._parameter_specs: list[ParameterSpec] | None = None
        self._group_specs: list[ParameterGroupSpec] | None = None

    @property
    def plan(self) -> OptimizationPlan:
        if self._plan is None:
            self._plan = self.build_plan()
        return self._plan

    def optimizer_count(self) -> int:
        return len(self._optimizer_groups(self._get_group_specs()))

    def build_plan(self) -> OptimizationPlan:
        specs = self._get_parameter_specs()
        groups = self._get_group_specs()
        optimizers, schedulers, parameter_to_optimizer = self._materialize(
            groups
        )
        plan = OptimizationPlan(
            optimizers=optimizers,
            schedulers=schedulers,
            parameter_groups=groups,
            parameter_to_optimizer=parameter_to_optimizer,
            frozen_specs=[
                spec for spec in specs if spec.frozen_until_epoch is not None
            ],
            uses_manual_optimization=self._uses_manual_optimization(
                optimizers
            ),
            strategy=self.training_strategy,
        )
        self._log_summary(plan)
        return plan

    def _get_parameter_specs(self) -> list[ParameterSpec]:
        if self._parameter_specs is None:
            self._parameter_specs = self._collect_parameter_specs()
        return self._parameter_specs

    def _get_group_specs(self) -> list[ParameterGroupSpec]:
        if self._group_specs is not None:
            return self._group_specs
        base_optimizer, base_scheduler = self._base_configs()
        groups = self._build_group_specs(
            self._get_parameter_specs(), base_optimizer, base_scheduler
        )

        if self.training_strategy is not None:
            self._validate_strategy_overrides(groups, base_optimizer)
            groups = self.training_strategy.transform_groups(groups)

        self._validate_parameter_ownership(groups)
        self._group_specs = groups
        return groups

    def _base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]:
        if self.training_strategy is None:
            return self.cfg.trainer.optimizer, self.cfg.trainer.scheduler
        return self.training_strategy.get_base_configs()

    def _collect_parameter_specs(self) -> list[ParameterSpec]:
        specs = []
        for node_name, node in self.nodes.items():
            for module_name, module in node.module.named_modules():
                if not list(module.parameters(recurse=False)):
                    continue
                for parameter_name, parameter in module.named_parameters(
                    recurse=False
                ):
                    specs.append(
                        ParameterSpec(
                            name=self._parameter_match_name(
                                module, module_name, parameter_name
                            ),
                            parameter=parameter,
                            node_name=node_name,
                            formatted_node_name=node.formatted_name,
                            module_name=module_name,
                            module_type=module.__class__.__name__,
                            parameter_name=parameter_name,
                            role=self._parameter_role(
                                module, parameter_name
                            ),
                            requires_grad=parameter.requires_grad,
                            frozen_until_epoch=getattr(
                                node, "unfreeze_after", None
                            ),
                        )
                    )
        return specs

    def _build_group_specs(
        self,
        specs: Sequence[ParameterSpec],
        base_optimizer: OptimizerConfig,
        base_scheduler: SchedulerConfig,
    ) -> list[ParameterGroupSpec]:
        grouped_specs: dict[
            tuple[
                str,
                tuple[tuple[str, Any], ...],
                str,
                tuple[tuple[str, Any], ...],
                str,
            ],
            ParameterGroupSpec,
        ] = {}

        for spec in specs:
            optimizer_cfg = base_optimizer.to_finetuning()
            scheduler_cfg = base_scheduler.to_finetuning()
            source = "trainer.optimizer"
            best_specificity = -1
            best_order = -1

            node = self.nodes[spec.node_name]
            for order, finetuning in enumerate(node.cfg.finetuning):
                specificity = self._selector_specificity(finetuning, spec)
                if specificity < 0:
                    continue
                if (specificity, order) < (best_specificity, best_order):
                    continue
                optimizer_cfg = merge_config_items(
                    base_optimizer, finetuning.optimizer
                )
                scheduler_cfg = merge_config_items(
                    base_scheduler, finetuning.scheduler
                )
                source = f"node:{spec.node_name}.finetuning[{order}]"
                best_specificity = specificity
                best_order = order

            key = (
                optimizer_cfg.name,
                self._hashable_params(optimizer_cfg.params),
                scheduler_cfg.name,
                self._hashable_params(scheduler_cfg.params),
                source,
            )
            if key not in grouped_specs:
                grouped_specs[key] = ParameterGroupSpec(
                    params=[],
                    optimizer_name=optimizer_cfg.name,
                    optimizer_params=dict(optimizer_cfg.params),
                    scheduler_name=scheduler_cfg.name,
                    scheduler_params=dict(scheduler_cfg.params),
                    source=source,
                    tags={spec.node_name},
                )
            grouped_specs[key].params.append(spec.parameter)
            grouped_specs[key].tags.add(spec.role)

        return list(grouped_specs.values())

    def _materialize(
        self, groups: Sequence[ParameterGroupSpec]
    ) -> tuple[
        list[Optimizer],
        list[LRSchedulerTypeUnion | LRSchedulerConfig],
        dict[int, int],
    ]:
        optimizer_groups = self._optimizer_groups(groups)

        optimizers: list[Optimizer] = []
        schedulers: list[LRSchedulerTypeUnion | LRSchedulerConfig] = []
        parameter_to_optimizer: dict[int, int] = {}

        for optimizer_idx, (
            (optimizer_name, scheduler_name, _),
            group_specs,
        ) in enumerate(optimizer_groups.items()):
            param_groups = []
            for group in group_specs:
                if not group.params:
                    continue
                param_group = {"params": group.params} | group.optimizer_params
                if group.strategy_role is not None:
                    param_group["strategy_role"] = group.strategy_role
                param_groups.append(param_group)
            cfg_optimizer = OptimizerConfig(
                name=optimizer_name,
                params={"params": param_groups},
            )
            cfg_scheduler = SchedulerConfig(
                name=scheduler_name,
                params=dict(group_specs[0].scheduler_params),
            )
            optimizer, scheduler = build_optimizer_scheduler(
                self.cfg,
                self._formatted_main_metric(),
                cfg_optimizer,
                cfg_scheduler,
            )
            if self.training_strategy is not None:
                scheduler = self.training_strategy.create_scheduler(
                    optimizer, group_specs, scheduler
                )
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            for group in group_specs:
                for parameter in group.params:
                    parameter_to_optimizer[id(parameter)] = optimizer_idx

        return optimizers, schedulers, parameter_to_optimizer

    def _optimizer_groups(
        self, groups: Sequence[ParameterGroupSpec]
    ) -> dict[
        tuple[str, str, tuple[tuple[str, Any], ...]], list[ParameterGroupSpec]
    ]:
        optimizer_groups: dict[
            tuple[str, str, tuple[tuple[str, Any], ...]],
            list[ParameterGroupSpec],
        ] = defaultdict(list)
        for group in groups:
            optimizer_groups[
                (
                    group.optimizer_name,
                    group.scheduler_name,
                    self._hashable_params(group.scheduler_params),
                )
            ].append(group)
        return optimizer_groups

    def _formatted_main_metric(self) -> MainMetric | None:
        if self.main_metric is None:
            return None
        return MainMetric(
            self.nodes.formatted_name(self.main_metric.node_name),
            self.main_metric.metric_name,
        )

    def _validate_strategy_overrides(
        self,
        groups: Sequence[ParameterGroupSpec],
        base_optimizer: OptimizerConfig,
    ) -> None:
        for group in groups:
            if group.optimizer_name == base_optimizer.name:
                continue
            if self.training_strategy.supports_optimizer_override(
                group.optimizer_name
            ):
                continue
            strategy_name = type(self.training_strategy).__name__
            raise ValueError(
                f"{strategy_name} only supports "
                f"{base_optimizer.name}-compatible "
                "finetuning overrides. Switching optimizer class to "
                f"'{group.optimizer_name}' is not supported."
            )

    def _validate_parameter_ownership(
        self, groups: Sequence[ParameterGroupSpec]
    ) -> None:
        seen: dict[int, str] = {}
        for group in groups:
            for parameter in group.params:
                parameter_id = id(parameter)
                if parameter_id in seen:
                    raise ValueError(
                        "Parameter is assigned to multiple optimizer groups: "
                        f"{seen[parameter_id]} and {group.source}."
                    )
                seen[parameter_id] = group.source

    def _uses_manual_optimization(
        self, optimizers: Sequence[Optimizer]
    ) -> bool:
        if self.training_strategy is not None:
            return self.training_strategy.requires_manual_optimization(
                len(optimizers)
            )
        return len(optimizers) > 1

    def _selector_specificity(
        self, finetuning: FinetuningConfig, spec: ParameterSpec
    ) -> int:
        if not finetuning.parameters:
            return 0
        scores = [
            self._pattern_specificity(pattern, spec)
            for pattern in finetuning.parameters
        ]
        return max(scores, default=-1)

    def _pattern_specificity(
        self, pattern: ParameterPattern, spec: ParameterSpec
    ) -> int:
        name_matches = (
            pattern.name is not None
            and pattern.name.lower() in spec.name.lower()
        )
        module_matches = (
            pattern.module_type is not None
            and pattern.module_type.lower() == spec.module_type.lower()
        )
        if pattern.name is not None and pattern.module_type is not None:
            return 30 if name_matches and module_matches else -1
        if pattern.name is not None:
            return 20 if name_matches else -1
        if pattern.module_type is not None:
            return 10 if module_matches else -1
        return -1

    @staticmethod
    def _parameter_match_name(
        module: nn.Module, module_name: str, parameter_name: str
    ) -> str:
        return f"{module.__class__.__name__}.{module_name}.{parameter_name}"

    @staticmethod
    def _parameter_role(
        module: nn.Module, parameter_name: str
    ) -> Literal["weight", "bias", "batch_norm_weight", "other"]:
        if parameter_name == "bias":
            return "bias"
        if parameter_name == "weight":
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                return "batch_norm_weight"
            return "weight"
        return "other"

    @staticmethod
    def _hashable_params(
        params: dict[str, Any],
    ) -> tuple[tuple[str, Any], ...]:
        return tuple(
            sorted((key, repr(value)) for key, value in params.items())
        )

    def _log_summary(self, plan: OptimizationPlan) -> None:
        logger.info(
            f"Optimization plan uses {len(plan.optimizers)} optimizer(s)."
        )
        for idx, optimizer in enumerate(plan.optimizers):
            logger.info(
                f"Optimizer {idx}: {type(optimizer).__name__}, "
                f"{len(optimizer.param_groups)} parameter group(s)."
            )
        for group in plan.parameter_groups:
            logger.info(
                "Optimization group: "
                f"source={group.source}, optimizer={group.optimizer_name}, "
                f"scheduler={group.scheduler_name}, "
                f"params={len(group.params)}, "
                f"tags={sorted(group.tags)}, role={group.strategy_role}"
            )
