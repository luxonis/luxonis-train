"""Resolution of node ``finetuning`` rules and node freezing into a
single optimizer configuration.

The pipeline has two phases:

    1. L{resolve_training_plan}: a pure function turning the config and
       the built nodes into a L{TrainingPlan} - a total, static
       partition of every model parameter (frozen parameters included)
       into parameter groups, each owned by exactly one inner
       optimizer/scheduler specification. Nothing torch-stateful is
       created here and the function can be called any number of times.
    2. L{build_training_plan}: instantiates the inner optimizers and
       their member schedulers from the plan. With a single inner the
       raw optimizer and scheduler are returned in the exact shapes
       Lightning saw before this module existed (so plain configs keep
       byte-identical checkpoints); with several inners everything is
       wrapped into one L{CompositeOptimizer} plus composite scheduler
       wrappers so the model stays in automatic optimization.

Because the partition is total, unfreezing never changes group
membership - it is purely a ``requires_grad`` flip (torch optimizers
skip parameters whose gradient is ``None``), which makes
checkpoint-resume a plain ``state_dict`` round trip.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, overload

from loguru import logger
from luxonis_ml.typing import Params
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    ReduceLROnPlateau,
    SequentialLR,
)

from luxonis_train.config import Config
from luxonis_train.config.config import (
    FinetuningOptimizerConfig,
    FinetuningSchedulerConfig,
    OptimizerConfig,
    ParameterPattern,
    SchedulerConfig,
)
from luxonis_train.optimizers.composite_optimizer import (
    CompositeOptimizer,
    unwrap_optimizers,
)
from luxonis_train.registry import OPTIMIZERS, SCHEDULERS, from_registry
from luxonis_train.schedulers.composite_scheduler import (
    CompositeLRScheduler,
    CompositeReduceLROnPlateau,
    rebase_scheduler_lr,
)

if TYPE_CHECKING:
    from luxonis_train.lightning.utils import Nodes, NodeWrapper
    from luxonis_train.strategies import BaseTrainingStrategy

__all__ = [
    "GroupHandle",
    "GroupSpec",
    "InnerSpec",
    "Selector",
    "StrategyRule",
    "TrainingPlan",
    "TrainingPlanRuntime",
    "build_training_plan",
    "merge_config_items",
    "pattern_selector",
    "resolve_training_plan",
    "unwrap_optimizers",
]

_SHARED = "__shared__"


class Selector(Protocol):
    """Predicate deciding whether a rule claims a parameter."""

    def __call__(
        self,
        module: nn.Module,
        module_name: str,
        parameter: nn.Parameter,
        parameter_name: str,
    ) -> bool: ...


def pattern_selector(patterns: Sequence[ParameterPattern]) -> Selector:
    """Adapt YAML ``parameters`` patterns to the L{Selector} protocol,
    preserving their documented semantics (unanchored, case-insensitive
    C{re.search} on the dotted parameter name and the module class
    name).
    """

    def match(
        module: nn.Module,
        module_name: str,
        parameter: nn.Parameter,
        parameter_name: str,
    ) -> bool:
        _ = parameter
        name = (
            f"{module_name}.{parameter_name}"
            if module_name
            else parameter_name
        )
        return any(
            pattern.matches(type(module).__name__, name)
            for pattern in patterns
        )

    return match


def _match_all(
    module: nn.Module,
    module_name: str,
    parameter: nn.Parameter,
    parameter_name: str,
) -> bool:
    _ = module, module_name, parameter, parameter_name
    return True


@overload
def merge_config_items(
    base: OptimizerConfig,
    override: FinetuningOptimizerConfig | None,
) -> OptimizerConfig: ...


@overload
def merge_config_items(
    base: SchedulerConfig,
    override: FinetuningSchedulerConfig | None,
) -> SchedulerConfig: ...


def merge_config_items(
    base: OptimizerConfig | SchedulerConfig,
    override: FinetuningOptimizerConfig | FinetuningSchedulerConfig | None,
) -> OptimizerConfig | SchedulerConfig:
    if override is None:
        # Not `base.to_finetuning()`: that returns a `Finetuning*Config`,
        # whose `name` is `str | None` because it models a *partial*
        # user override. Callers here require a concrete name.
        return type(base)(name=base.name, params=base.params)

    if override.name is None or override.name == base.name:
        name = base.name
        params = base.params | override.params
    else:
        name = override.name
        params = override.params

    return type(base)(name=name, params=params)


def _spec_key(name: str, params: Params) -> str:
    return json.dumps(
        {"name": name, "params": params}, sort_keys=True, default=repr
    )


@dataclass(frozen=True)
class OptimizerSpec:
    """Canonical optimizer specification of one rule.

    C{params} double as the parameter-group options of the groups the
    rule produces (matching the previous behavior, where per-group
    hyperparameters carried the full optimizer configuration).
    """

    name: str
    params: Params

    @classmethod
    def from_config(cls, config: OptimizerConfig) -> "OptimizerSpec":
        OPTIMIZERS.get(config.name)  # fail early on unknown names
        return cls(name=config.name, params=dict(config.params))


@dataclass(frozen=True)
class SchedulerSpec:
    name: str
    params: Params
    key: str = field(compare=False)

    @classmethod
    def from_config(
        cls, config: SchedulerConfig, total_epochs: int
    ) -> "SchedulerSpec":
        SCHEDULERS.get(config.name)  # fail early on unknown names
        params = dict(config.params)
        # Defaults are injected *before* grouping keys are computed, so
        # two rules that both omit `T_max` collapse into one optimizer.
        if config.name == "CosineAnnealingLR":
            if "T_max" not in params:
                params["T_max"] = total_epochs
                logger.warning(
                    "`T_max` was not set for 'CosineAnnealingLR' "
                    "Automatically setting `T_max` to number of epochs."
                )
            elif params["T_max"] != total_epochs:
                logger.warning(
                    "Parameter `T_max` of 'CosineAnnealingLR' is "
                    "not equal to the number of epochs. "
                    "Make sure this is intended."
                    f"`T_max`: {params['T_max']}, "
                    f"Number of epochs: {total_epochs}"
                )
        return cls(
            name=config.name,
            params=params,
            key=_spec_key(config.name, params),
        )


@dataclass(frozen=True)
class StrategyRule:
    """A parameter-group rule contributed by a training strategy.

    Evaluated after every node ``finetuning`` rule and before the
    default tail. C{scheduler=None} inherits the strategy's base
    scheduler. The C{tag} keys the group handles the strategy receives
    back through C{attach} after the optimizers are built.
    """

    tag: str
    selector: Selector
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = None


@dataclass(frozen=True)
class Rule:
    """One parameter-claiming rule of the partition."""

    source: str  # "node" | "strategy" | "default"
    label: str
    selector: Selector
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec
    tag: str | None = None


class GroupHandle(NamedTuple):
    """Stable address of one parameter group.

    Index-based on purpose: ``Optimizer.load_state_dict`` replaces the
    group dictionaries on checkpoint restore, but with a static
    partition the indices never move.
    """

    inner_index: int
    group_index: int


@dataclass(frozen=True)
class GroupSpec:
    name: str
    node_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]
    parameter_names: tuple[str, ...]
    options: Params


@dataclass(frozen=True)
class InnerSpec:
    optimizer_name: str
    scheduler: SchedulerSpec
    groups: tuple[GroupSpec, ...]


@dataclass(frozen=True)
class TrainingPlan:
    """Total static partition of the model parameters."""

    inners: tuple[InnerSpec, ...]
    handles_by_node: Mapping[str, tuple[GroupHandle, ...]]
    handles_by_tag: Mapping[str, tuple[GroupHandle, ...]]

    def __post_init__(self) -> None:
        seen: set[int] = set()
        for inner in self.inners:
            for group in inner.groups:
                for parameter in group.parameters:
                    if id(parameter) in seen:  # pragma: no cover
                        raise RuntimeError(
                            "Internal error: parameter "
                            "claimed by more than one group."
                        )
                    seen.add(id(parameter))


class _GroupDraft:
    def __init__(self, name: str, options: Params, tag: str | None):
        self.name = name
        self.options = options
        self.tag = tag
        self.node_names: list[str] = []
        self.parameters: list[nn.Parameter] = []
        self.parameter_names: list[str] = []

    def add(
        self, node_name: str, parameter: nn.Parameter, parameter_name: str
    ) -> None:
        if node_name not in self.node_names:
            self.node_names.append(node_name)
        self.parameters.append(parameter)
        self.parameter_names.append(parameter_name)


class _PlanBuilder:
    def __init__(self) -> None:
        # inner key -> (optimizer name, scheduler spec)
        self._inners: dict[tuple[str, str], tuple[str, SchedulerSpec]] = {}
        # (inner key, group key) -> draft
        self._groups: dict[
            tuple[tuple[str, str], tuple[str, str]], _GroupDraft
        ] = {}
        self._claimed: set[int] = set()

    @property
    def claimed_ids(self) -> set[int]:
        return set(self._claimed)

    def mark_claimed(self, parameter_ids: set[int]) -> None:
        self._claimed |= parameter_ids

    def claim(
        self,
        rule: Rule,
        node_name: str,
        module_source: nn.Module,
        group_scope: str,
    ) -> int:
        """Claim all yet-unclaimed parameters of C{module_source}
        matched by the rule into the rule's group, returning how many
        parameters were claimed.
        """
        inner_key = (rule.optimizer.name, rule.scheduler.key)
        group_key = (rule.label, group_scope)
        claimed = 0
        for module_name, module in module_source.named_modules():
            for parameter_name, parameter in module.named_parameters(
                recurse=False
            ):
                if id(parameter) in self._claimed:
                    continue
                if not rule.selector(
                    module, module_name, parameter, parameter_name
                ):
                    continue
                self._claimed.add(id(parameter))
                if inner_key not in self._inners:
                    self._inners[inner_key] = (
                        rule.optimizer.name,
                        rule.scheduler,
                    )
                draft = self._groups.get((inner_key, group_key))
                if draft is None:
                    draft = _GroupDraft(
                        name=rule.label
                        if group_scope == _SHARED
                        else f"{rule.label}/{group_scope}",
                        options=dict(rule.optimizer.params),
                        tag=rule.tag,
                    )
                    self._groups[inner_key, group_key] = draft
                full_name = (
                    f"{module_name}.{parameter_name}"
                    if module_name
                    else parameter_name
                )
                draft.add(node_name, parameter, f"{node_name}.{full_name}")
                claimed += 1
        return claimed

    def finish(self) -> TrainingPlan:
        inners: list[InnerSpec] = []
        handles_by_node: dict[str, list[GroupHandle]] = {}
        handles_by_tag: dict[str, list[GroupHandle]] = {}
        for inner_index, (inner_key, (optimizer_name, scheduler)) in enumerate(
            self._inners.items()
        ):
            groups: list[GroupSpec] = []
            for (key, _), draft in self._groups.items():
                if key != inner_key:
                    continue
                handle = GroupHandle(inner_index, len(groups))
                groups.append(
                    GroupSpec(
                        name=draft.name,
                        node_names=tuple(draft.node_names),
                        parameters=tuple(draft.parameters),
                        parameter_names=tuple(draft.parameter_names),
                        options=draft.options,
                    )
                )
                for node_name in draft.node_names:
                    handles_by_node.setdefault(node_name, []).append(handle)
                if draft.tag is not None:
                    handles_by_tag.setdefault(draft.tag, []).append(handle)
            inners.append(
                InnerSpec(
                    optimizer_name=optimizer_name,
                    scheduler=scheduler,
                    groups=tuple(groups),
                )
            )
        return TrainingPlan(
            inners=tuple(inners),
            handles_by_node={
                name: tuple(handles)
                for name, handles in handles_by_node.items()
            },
            handles_by_tag={
                tag: tuple(handles) for tag, handles in handles_by_tag.items()
            },
        )


def resolve_training_plan(
    cfg: Config,
    nodes: "Nodes",
    strategy: "BaseTrainingStrategy | None" = None,
) -> TrainingPlan:
    """Resolve node ``finetuning`` rules, strategy rules, node freezing,
    and the trainer-level optimizer/scheduler into a total static
    parameter partition.

    Claiming is first-match-wins: a node's own rules (in YAML order)
    are tried first, then the strategy's rules, then the default tail.
    Nodes are visited in graph order and parameters in module-traversal
    order, so the partition is deterministic. Every parameter - frozen
    parameters included - ends up in exactly one group, which is what
    guarantees that a frozen node always has an optimizer to train it
    after unfreezing.

    With a strategy, the strategy's base configs are the inheritance
    base for node rules and the specification of the default tail.
    """
    epochs = cfg.trainer.epochs
    if strategy is not None:
        try:
            base_optimizer, base_scheduler = strategy.get_base_configs()
        except NotImplementedError:
            base_optimizer = cfg.trainer.optimizer
            base_scheduler = cfg.trainer.scheduler
    else:
        base_optimizer = cfg.trainer.optimizer
        base_scheduler = cfg.trainer.scheduler
    tail_optimizer = OptimizerSpec.from_config(base_optimizer)
    tail_scheduler = SchedulerSpec.from_config(base_scheduler, epochs)

    builder = _PlanBuilder()
    any_node_rules = any(node.finetuning for node in nodes.values())

    tail = Rule(
        source="default",
        label="default",
        selector=_match_all,
        optimizer=tail_optimizer,
        scheduler=tail_scheduler,
    )

    def tail_scope(node: "NodeWrapper") -> str:
        # Freezing-scheduled nodes get their own tail group so that
        # `lr_after_unfreeze` has a well-scoped target (the node-purity
        # invariant); everything else shares one group, or one group
        # per node when finetuning rules are present without a strategy
        # (preserving the previous observable grouping).
        if node.unfreeze_after is not None:
            return node.name
        if any_node_rules and strategy is None:
            return node.name
        return _SHARED

    for node in nodes.values():
        for index, finetuning in enumerate(node.finetuning):
            rule = Rule(
                source="node",
                label=f"{node.name}/{index}",
                selector=pattern_selector(
                    finetuning.parameters or [ParameterPattern(name=".*")]
                ),
                optimizer=OptimizerSpec.from_config(
                    merge_config_items(base_optimizer, finetuning.optimizer)
                ),
                scheduler=SchedulerSpec.from_config(
                    merge_config_items(base_scheduler, finetuning.scheduler),
                    epochs,
                ),
            )
            if not builder.claim(rule, node.name, node.module, _SHARED):
                raise ValueError(
                    "Finetuning parameters for node "
                    f"'{node.name}' did not match any "
                    "available trainable parameters."
                )
        if strategy is None:
            # Running the tail inside the node loop preserves the group
            # and optimizer creation order of the previous
            # implementation.
            builder.claim(tail, node.name, node.module, tail_scope(node))

    if strategy is not None:
        opaque_ids = strategy.opaque_parameter_ids()
        overlap = opaque_ids & builder.claimed_ids
        if overlap:
            name = getattr(strategy, "legacy_name", type(strategy).__name__)
            raise ValueError(
                f"Legacy training strategy '{name}' claims "
                f"{len(overlap)} parameter(s) already claimed by "
                "finetuning rules. Remove the overlapping rules or "
                "port the strategy to the new `rules()` API."
            )
        builder.mark_claimed(opaque_ids)
        for strategy_rule in strategy.rules():
            rule = Rule(
                source="strategy",
                label=f"strategy/{strategy_rule.tag}",
                selector=strategy_rule.selector,
                optimizer=OptimizerSpec.from_config(strategy_rule.optimizer),
                scheduler=SchedulerSpec.from_config(
                    strategy_rule.scheduler
                    if strategy_rule.scheduler is not None
                    else base_scheduler,
                    epochs,
                ),
                tag=strategy_rule.tag,
            )
            for node in nodes.values():
                scope = (
                    node.name if node.unfreeze_after is not None else _SHARED
                )
                builder.claim(rule, node.name, node.module, scope)
        for node in nodes.values():
            builder.claim(tail, node.name, node.module, tail_scope(node))

    return builder.finish()


def _build_member_scheduler(
    spec: SchedulerSpec, optimizer: Optimizer
) -> LRScheduler:
    def get(config: SchedulerConfig) -> LRScheduler:
        return from_registry(
            SCHEDULERS, config.name, **config.params, optimizer=optimizer
        )

    if spec.name == "SequentialLR":
        sequential = SchedulerConfig(
            name=spec.name, params=spec.params
        ).get_sequential_lr_params()
        return SequentialLR(
            optimizer,
            schedulers=[get(child) for child in sequential.schedulers],
            milestones=sequential.milestones,
            last_epoch=sequential.last_epoch,
        )
    return from_registry(
        SCHEDULERS, spec.name, **spec.params, optimizer=optimizer
    )


def _plateau_monitor(
    spec: SchedulerSpec, main_metric_monitor: str | None
) -> str:
    if spec.params.get("mode") == "max":
        if main_metric_monitor is None:
            raise ValueError(
                "ReduceLROnPlateau with 'max' mode "
                "requires a metric to monitor."
            )
        return main_metric_monitor
    return "val/loss"


@dataclass
class TrainingPlanRuntime:
    """The built optimizers and schedulers of a L{TrainingPlan},
    together with the handle-based accessors the freezing subsystem
    uses.
    """

    plan: TrainingPlan
    inner_optimizers: tuple[Optimizer, ...]
    members: tuple[LRScheduler | ReduceLROnPlateau | None, ...]
    optimizer: Optimizer
    scheduler_configs: list[Any]

    def group(self, handle: GroupHandle) -> dict[str, Any]:
        optimizer = self.inner_optimizers[handle.inner_index]
        return optimizer.param_groups[handle.group_index]

    def handles_for_node(self, node_name: str) -> tuple[GroupHandle, ...]:
        return self.plan.handles_by_node.get(node_name, ())

    def set_group_base_lr(self, handle: GroupHandle, lr: float) -> None:
        """Rebase a group's learning rate so it survives future
        scheduler steps: updates the group's ``lr`` and ``initial_lr``
        and the owning member scheduler's ``base_lrs`` entry (recursing
        into ``SequentialLR`` children).
        """
        group = self.group(handle)
        group["lr"] = float(lr)
        group["initial_lr"] = float(lr)
        member = self.members[handle.inner_index]
        if member is not None:
            rebase_scheduler_lr(member, handle.group_index, lr)


def build_training_plan(
    plan: TrainingPlan,
    cfg: Config,
    main_metric_monitor: str | None,
    strategy: "BaseTrainingStrategy | None" = None,
) -> TrainingPlanRuntime:
    """Instantiate the optimizers and schedulers of a plan.

    A single inner is returned raw (the exact shapes Lightning received
    before this module existed); several inners are wrapped into one
    L{CompositeOptimizer} plus per-bucket composite schedulers so the
    model stays in automatic optimization. Optimizers mounted by a
    legacy strategy adapter are appended as additional (opaque) inners.
    """
    inner_optimizers: list[Optimizer] = []
    # one entry per inner: (member scheduler or None, plateau monitor
    # or None)
    entries: list[
        tuple[LRScheduler | ReduceLROnPlateau | None, str | None]
    ] = []
    bypass_configs: list[Any] | None = None
    for inner in plan.inners:
        torch_groups = [
            {"params": list(group.parameters), **group.options}
            for group in inner.groups
        ]
        optimizer = from_registry(
            OPTIMIZERS, inner.optimizer_name, params=torch_groups
        )
        optimizer_keys = set(optimizer.defaults) | {"params"}
        for group in optimizer.param_groups:
            unknown_keys = set(group) - optimizer_keys
            if unknown_keys:
                keys = ", ".join(sorted(unknown_keys))
                raise TypeError(
                    f"Invalid parameter group option(s) for optimizer "
                    f"'{inner.optimizer_name}': {keys}"
                )
        inner_optimizers.append(optimizer)
        member = _build_member_scheduler(inner.scheduler, optimizer)
        if isinstance(member, ReduceLROnPlateau):
            entries.append(
                (
                    member,
                    _plateau_monitor(inner.scheduler, main_metric_monitor),
                )
            )
        else:
            entries.append((member, None))

    for optimizer, scheduler in (
        strategy.opaque_inners() if strategy is not None else []
    ):
        inner_optimizers.append(optimizer)
        if isinstance(scheduler, dict):
            entries.append((scheduler["scheduler"], scheduler.get("monitor")))
            if bypass_configs is None:
                bypass_configs = [scheduler]
        elif scheduler is None:
            entries.append((None, None))
        else:
            entries.append((scheduler, None))
            if bypass_configs is None:
                bypass_configs = [scheduler]

    validation_interval = cfg.trainer.validation_interval

    if len(inner_optimizers) == 1:
        if bypass_configs is None:
            member, monitor = entries[0]
            if monitor is not None:
                bypass_configs = [
                    {
                        "scheduler": member,
                        "monitor": monitor,
                        "frequency": validation_interval,
                    }
                ]
            elif member is not None:
                bypass_configs = [member]
            else:  # pragma: no cover
                bypass_configs = []
        return TrainingPlanRuntime(
            plan=plan,
            inner_optimizers=tuple(inner_optimizers),
            members=tuple(entry[0] for entry in entries),
            optimizer=inner_optimizers[0],
            scheduler_configs=bypass_configs,
        )

    composite = CompositeOptimizer(inner_optimizers)
    epoch_members = [
        member
        for member, monitor in entries
        if member is not None and monitor is None
    ]
    plateau_by_monitor: dict[str, list[ReduceLROnPlateau]] = {}
    for member, monitor in entries:
        if member is not None and monitor is not None:
            assert isinstance(member, ReduceLROnPlateau)
            plateau_by_monitor.setdefault(monitor, []).append(member)

    scheduler_configs: list[Any] = []
    if epoch_members:
        scheduler_configs.append(
            {
                "scheduler": CompositeLRScheduler(composite, epoch_members),
                "name": "lr",
            }
        )
    for index, (monitor, plateau_members) in enumerate(
        plateau_by_monitor.items()
    ):
        name = "lr-plateau" if index == 0 else f"lr-plateau-{index}"
        scheduler_configs.append(
            {
                "scheduler": CompositeReduceLROnPlateau(
                    composite, plateau_members
                ),
                "monitor": monitor,
                "frequency": validation_interval,
                "reduce_on_plateau": True,
                "name": name,
            }
        )

    return TrainingPlanRuntime(
        plan=plan,
        inner_optimizers=tuple(inner_optimizers),
        members=tuple(entry[0] for entry in entries),
        optimizer=composite,
        scheduler_configs=scheduler_configs,
    )
