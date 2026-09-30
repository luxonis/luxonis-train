"""The partition of the model parameters into optimizer groups.

The module turns the node ``finetuning`` entries, the rules of a
training strategy, and the node freezing into one optimizer
configuration. It works in two phases:

1. `resolve_training_plan` builds a `TrainingPlan` from the config and
   the nodes. The plan holds every parameter of every node in exactly
   one parameter group, frozen parameters included. The parameters of
   a legacy training strategy are the only exception. Each group
   belongs to one inner optimizer and its scheduler. The function does
   not create the optimizers or the schedulers of the plan.
2. `build_training_plan` creates the inner optimizers and their member
   schedulers from the plan. With one inner optimizer, Lightning
   receives that optimizer and its scheduler directly, so the checkpoint
   of a plain config holds a plain optimizer state. With several inner
   optimizers, one `CompositeOptimizer` and composite schedulers wrap
   them, so the model stays in the automatic optimization of Lightning.

The partition is static. When a node unfreezes, its parameters stay in
their groups. Only ``requires_grad`` changes, and a torch optimizer
skips a parameter whose gradient is ``None``. A resume from a checkpoint
is therefore a plain ``state_dict`` round trip.

"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, overload

from lightning.pytorch.utilities.types import LRSchedulerConfigType
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

# One entry per inner optimizer: (member scheduler or None, plateau
# monitor or None).
_MemberEntry = tuple[LRScheduler | ReduceLROnPlateau | None, str | None]
_BypassConfig = LRSchedulerConfigType | LRScheduler | ReduceLROnPlateau


class Selector(Protocol):
    """A predicate that decides whether a rule claims a parameter.

    `pattern_selector` builds a selector from the ``parameters``
    patterns of a node ``finetuning`` entry. A training strategy gives
    its own selector in each `StrategyRule`. A selector sees only the
    parameters that no earlier rule claimed.

    """

    def __call__(
        self,
        module: nn.Module,
        module_name: str,
        parameter: nn.Parameter,
        parameter_name: str,
    ) -> bool:
        """Decide whether the rule claims the parameter.

        Args:
            module (torch.nn.Module): The module that directly owns the
                parameter.
            module_name (str): The dotted name of ``module`` relative to
                the node, such as ``"stem.conv"``. It is an empty string
                for the node module itself.
            parameter (``nn.Parameter``): The parameter.
            parameter_name (str): The name of the parameter inside
                ``module``, such as ``"weight"``.

        Returns:
            bool: ``True`` when the rule claims the parameter.

        """
        ...


def pattern_selector(patterns: Sequence[ParameterPattern]) -> Selector:
    """Build a `Selector` from the patterns of a ``finetuning`` entry.

    The selector joins ``module_name`` and ``parameter_name`` into the
    dotted parameter name, such as ``"stem.conv.weight"``. When
    ``module_name`` is empty, the dotted name is ``parameter_name``
    alone. The selector accepts the parameter when at least one pattern
    matches. A pattern matches when its ``name`` matches the dotted name
    and its ``module_type`` matches the class name of ``module``. A
    field left as ``None`` matches everything. Both fields are regular
    expressions. ``re.search`` matches them without anchors and without
    case, as `ParameterPattern` describes.

    Args:
        patterns (``Sequence[ParameterPattern]``): The patterns. An empty
            sequence gives a selector that accepts no parameter.

    Returns:
        Selector: A function with the arguments of `Selector.__call__`.
        It returns ``True`` when a pattern matches the parameter.

    Example:
        >>> from torch import nn
        >>> from luxonis_train.config.config import ParameterPattern
        >>> linear = nn.Linear(2, 2)
        >>> select = pattern_selector([ParameterPattern(name="head.weight")])
        >>> select(linear, "head", linear.weight, "weight")
        True
        >>> select(linear, "head", linear.bias, "bias")
        False

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
    """Merge a finetuning override into a base optimizer or scheduler.

    The merge follows these rules:

    - Without an override, the result is a copy of ``base``.
    - An override without a ``name``, or with the name of ``base``, keeps
      the name of ``base``. The result has the ``params`` of ``base``,
      updated with the ``params`` of the override.
    - An override with a different ``name`` replaces ``base``. The result
      has only the ``params`` of the override.

    Args:
        base (OptimizerConfig | SchedulerConfig): The base config:
            ``trainer.optimizer``, ``trainer.scheduler``, or a base
            config of the training strategy.
        override (FinetuningOptimizerConfig | FinetuningSchedulerConfig | None):
            The override of a node ``finetuning`` entry, or ``None``.

    Returns:
        OptimizerConfig | SchedulerConfig: A new config of the type of
        ``base``, so its ``name`` is never ``None``. The arguments do not
        change.

    Example:
        >>> from luxonis_train.config.config import (
        ...     FinetuningOptimizerConfig,
        ...     OptimizerConfig,
        ... )
        >>> base = OptimizerConfig(
        ...     name="SGD", params={"lr": 0.01, "momentum": 0.9}
        ... )
        >>> lower_lr = FinetuningOptimizerConfig(params={"lr": 0.001})
        >>> merge_config_items(base, lower_lr)
        OptimizerConfig(name='SGD', params={'lr': 0.001, 'momentum': 0.9})
        >>> adam = FinetuningOptimizerConfig(name="Adam", params={"lr": 0.001})
        >>> merge_config_items(base, adam)
        OptimizerConfig(name='Adam', params={'lr': 0.001})

    """
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


@dataclass(frozen=True)
class OptimizerSpec:
    """The optimizer of one rule.

    Rules with the same optimizer name and the same scheduler
    `SchedulerSpec.key` share one inner optimizer. The ``params`` do not
    affect that choice, because each group carries them as its own
    options.

    Attributes:
        name (str): The class name of the optimizer in the
            ``OPTIMIZERS`` registry.
        params (``Params``): The optimizer parameters, such as ``lr``.
            Each parameter group of the rule receives them as its
            options.

    """

    name: str
    params: Params

    @classmethod
    def from_config(cls, config: OptimizerConfig) -> "OptimizerSpec":
        """Create the specification from an optimizer config.

        Args:
            config (OptimizerConfig): The optimizer config.

        Returns:
            OptimizerSpec: The name of ``config`` and a shallow copy of
            its ``params``.

        Raises:
            KeyError: When the ``OPTIMIZERS`` registry has no optimizer
                with the name of ``config``.

        """
        OPTIMIZERS.get(config.name)  # fail early on unknown names
        return cls(name=config.name, params=dict(config.params))


@dataclass(frozen=True)
class SchedulerSpec:
    """The scheduler of one rule.

    Attributes:
        name (str): The class name of the scheduler in the
            ``SCHEDULERS`` registry.
        params (``Params``): The scheduler parameters.
            `SchedulerSpec.from_config` adds ``T_max`` when a
            ``CosineAnnealingLR`` config has no ``T_max``.
        key (str): A JSON string of ``name`` and ``params`` with sorted
            keys. A value that JSON cannot encode goes into the string
            as its ``repr``. Rules with the same optimizer name and the
            same ``key`` share one inner optimizer. The equality check
            of the dataclass ignores this field.

    """

    name: str
    params: Params
    key: str = field(compare=False)

    @classmethod
    def from_config(
        cls, config: SchedulerConfig, total_epochs: int
    ) -> "SchedulerSpec":
        """Create the specification from a scheduler config.

        For ``CosineAnnealingLR``, the method sets ``T_max`` to
        ``total_epochs`` when ``params`` has no ``T_max``, and logs a
        warning. It also logs a warning when ``T_max`` is not equal to
        ``total_epochs``. The method adds the default before it computes
        ``key``. A rule that omits ``T_max`` and a rule that sets it to
        ``total_epochs`` therefore get the same ``key``.

        Args:
            config (SchedulerConfig): The scheduler config.
            total_epochs (int): The number of training epochs, from
                ``trainer.epochs``.

        Returns:
            SchedulerSpec: The name of ``config``, a shallow copy of its
            ``params`` with the ``T_max`` default when the method adds
            one, and the ``key``.

        Raises:
            KeyError: When the ``SCHEDULERS`` registry has no scheduler
                with the name of ``config``.

        Example:
            >>> from luxonis_train.config.config import SchedulerConfig
            >>> step = SchedulerConfig(
            ...     name="StepLR", params={"step_size": 5, "gamma": 0.5}
            ... )
            >>> SchedulerSpec.from_config(step, total_epochs=10).key
            '{"name": "StepLR", "params": {"gamma": 0.5, "step_size": 5}}'

        """
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
    """A parameter-group rule that a training strategy contributes.

    `BaseTrainingStrategy.rules` returns these rules.
    `resolve_training_plan` evaluates them in order, after the
    ``finetuning`` entries of every node and before the default rule. A
    rule visits every node before the next rule starts. It claims each
    parameter that its ``selector`` accepts and that no earlier rule
    claimed.

    The groups of a rule are named ``strategy/<tag>``. A node with
    ``freezing.active`` gets its own group ``strategy/<tag>/<node>``.

    Attributes:
        tag (str): The name of the rule.
            `BaseTrainingStrategy.attach` receives the handles of the
            groups of the rule under this key. When the rule claims no
            parameter, the mapping has no entry for the tag.
        selector (Selector): The predicate that decides which parameters
            the rule claims.
        optimizer (OptimizerConfig): The optimizer of the claimed
            parameters. The plan uses it as it is and does not merge it
            with the base optimizer.
        scheduler (SchedulerConfig | None): The scheduler of the claimed
            parameters. ``None`` uses the base scheduler of the strategy,
            or ``trainer.scheduler`` when
            `BaseTrainingStrategy.get_base_configs` raises
            ``NotImplementedError``.

    """

    tag: str
    selector: Selector
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = None


@dataclass(frozen=True)
class Rule:
    """A rule that claims parameters into the groups of the plan.

    `resolve_training_plan` creates one rule for each node ``finetuning``
    entry, one rule for each `StrategyRule`, and one default rule that
    claims all parameters that are left.

    Attributes:
        label (str): The base name of the groups of the rule:
            ``<node>/<index>`` for a ``finetuning`` entry,
            ``strategy/<tag>`` for a strategy rule, and ``default`` for
            the default rule.
        selector (Selector): The predicate that decides which parameters
            the rule claims.
        optimizer (OptimizerSpec): The optimizer of the claimed
            parameters.
        scheduler (SchedulerSpec): The scheduler of the claimed
            parameters.
        tag (str | None): The tag of a strategy rule, or ``None`` for
            the other rules.

    """

    label: str
    selector: Selector
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec
    tag: str | None = None


class GroupHandle(NamedTuple):
    """The stable address of one parameter group.

    The handle holds indices, not the group dictionary.
    ``Optimizer.load_state_dict`` replaces the group dictionaries when a
    checkpoint loads. The partition is static, so the indices stay
    valid.

    Attributes:
        inner_index (int): The index of the inner optimizer in
            `TrainingPlan.inners` and in
            `TrainingPlanRuntime.inner_optimizers`.
        group_index (int): The index of the group in `InnerSpec.groups`
            and in the ``param_groups`` of the inner optimizer.

    """

    inner_index: int
    group_index: int


@dataclass(frozen=True)
class GroupSpec:
    """One parameter group of the plan.

    Attributes:
        name (str): The name of the group, unique in the whole plan. It
            is the label of the rule, followed by ``/<node>`` when
            `resolve_training_plan` gives a node its own group of the
            rule. A repeated name gets the suffix ``-2``, ``-3``, and so
            on.
        node_names (``tuple[str, ...]``): The names of the nodes that
            have parameters in the group, in the order of the first
            claim.
        parameters (``tuple[nn.Parameter, ...]``): The parameters of the
            group, in claim order.
        parameter_names (``tuple[str, ...]``): The name of each parameter,
            in the order of ``parameters``. A name is the node name, a
            dot, and the dotted name of the parameter in the node, such
            as ``"backbone.stem.conv.weight"``.
        options (``Params``): The parameter-group options, from the
            ``params`` of the optimizer of the rule.

    """

    name: str
    node_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]
    parameter_names: tuple[str, ...]
    options: Params


@dataclass(frozen=True)
class InnerSpec:
    """One inner optimizer of the plan and its scheduler.

    Attributes:
        optimizer_name (str): The class name of the optimizer in the
            ``OPTIMIZERS`` registry.
        scheduler (SchedulerSpec): The scheduler of the optimizer.
        groups (``tuple[GroupSpec, ...]``): The parameter groups of the
            optimizer, in creation order.

    """

    optimizer_name: str
    scheduler: SchedulerSpec
    groups: tuple[GroupSpec, ...]


@dataclass(frozen=True)
class TrainingPlan:
    """The static partition of the model parameters into groups.

    `resolve_training_plan` puts every parameter of every node into
    exactly one group, frozen parameters included. The only exception
    is a parameter that a legacy training strategy claims for its own
    optimizers.

    Attributes:
        inners (``tuple[InnerSpec, ...]``): The specifications of the
            inner optimizers, in creation order.
        handles_by_node (``Mapping[str, tuple[GroupHandle, ...]]``): For
            each node name, the handles of the groups that hold
            parameters of the node, in plan order. A node without a
            parameter in the plan has no entry.
        handles_by_tag (``Mapping[str, tuple[GroupHandle, ...]]``): For
            each strategy rule tag, the handles of the groups of the
            rule, in plan order. A tag whose rule claims no parameter
            has no entry.

    """

    inners: tuple[InnerSpec, ...]
    handles_by_node: Mapping[str, tuple[GroupHandle, ...]]
    handles_by_tag: Mapping[str, tuple[GroupHandle, ...]]

    def __post_init__(self) -> None:
        """Check that no parameter is in more than one group.

        Raises:
            RuntimeError: When a parameter object is in more than one
                group. This is an internal error.

        """
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
        """Claim the free parameters that a rule accepts from a module.

        Each claimed parameter goes into the group of ``rule`` for
        ``group_scope``. The method creates the group and the inner
        optimizer entry when they do not exist yet.

        Args:
            rule (Rule): The rule whose selector decides.
            node_name (str): The name of the node that owns
                ``module_source``.
            module_source (torch.nn.Module): The node module. The
                method scans it and all its submodules.
            group_scope (str): ``_SHARED`` for a group that all nodes
                share, or the node name for a group of this node only.

        Returns:
            int: The number of parameters that this call claimed. A
            parameter that an earlier call claimed does not count.

        """
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
                full_name = (
                    f"{module_name}.{parameter_name}"
                    if module_name
                    else parameter_name
                )
                draft = self._draft_for(rule, group_scope)
                draft.add(node_name, parameter, f"{node_name}.{full_name}")
                claimed += 1
        return claimed

    def finish(self) -> TrainingPlan:
        inners: list[InnerSpec] = []
        handles_by_node: dict[str, list[GroupHandle]] = {}
        handles_by_tag: dict[str, list[GroupHandle]] = {}
        # LearningRateMonitor requires names to be unique across the
        # composite optimizer, not just within each inner optimizer.
        used_names: set[str] = set()
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
                        name=_unique_name(draft.name, used_names),
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

    def _draft_for(self, rule: Rule, group_scope: str) -> _GroupDraft:
        inner_key = (rule.optimizer.name, rule.scheduler.key)
        if inner_key not in self._inners:
            self._inners[inner_key] = (rule.optimizer.name, rule.scheduler)
        group_key = (rule.label, group_scope)
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
        return draft


def resolve_training_plan(
    cfg: Config,
    nodes: "Nodes",
    strategy: "BaseTrainingStrategy | None" = None,
) -> TrainingPlan:
    """Resolve the parameter rules of a model into a `TrainingPlan`.

    The function combines the node ``finetuning`` entries, the rules of
    the training strategy, the node freezing, and the base optimizer and
    scheduler. It does not create the optimizers or the schedulers of
    the plan.

    The first rule that accepts a parameter claims it. The rules claim
    in this order:

    1. The ``finetuning`` entries of a node, in config order. An entry
       claims only parameters of its own node. An entry without
       ``parameters`` claims every free parameter of the node. Its
       optimizer and scheduler overrides merge into the base configs
       through `merge_config_items`.
    2. The rules of ``strategy``, in order. Each rule visits every node
       before the next rule starts. A legacy strategy claims its own
       parameters before these rules. Those parameters stay outside the
       plan.
    3. The default rule. It claims every parameter that is left, with
       the base optimizer and scheduler.

    Without a strategy, the default rule runs for each node directly
    after the ``finetuning`` entries of that node. With a strategy, the
    entries of all nodes run first, then the strategy rules, then the
    default rule. The function visits the nodes in the order of
    ``nodes`` and the parameters in module order, so the result is
    deterministic. The rules also claim frozen parameters, so a node
    that unfreezes already has an optimizer.

    The groups get these names:

    - ``<node>/<index>`` for a ``finetuning`` entry. ``<index>`` is the
      position of the entry in the ``finetuning`` list of the node,
      from ``0``.
    - ``strategy/<tag>`` for a strategy rule, shared by all nodes.
    - ``default`` for the default rule, shared by all nodes.
    - A node with ``freezing.active`` gets its own strategy and default
      groups, ``strategy/<tag>/<node>`` and ``default/<node>``. The
      ``lr_after_unfreeze`` rate of the node therefore changes no group
      of another node.
    - Without a strategy, when any node has ``finetuning`` entries,
      every node gets its own default group ``default/<node>``.

    Rules with the same optimizer name and the same `SchedulerSpec.key`
    share one inner optimizer.

    The base optimizer and scheduler are ``trainer.optimizer`` and
    ``trainer.scheduler``. With a strategy, they come from
    `BaseTrainingStrategy.get_base_configs`, unless that method raises
    ``NotImplementedError``.

    `SchedulerSpec.from_config` logs a warning for each
    ``CosineAnnealingLR`` rule whose ``T_max`` is missing or differs
    from ``trainer.epochs``.

    Args:
        cfg (Config): The config. The function reads ``trainer.epochs``,
            ``trainer.optimizer``, and ``trainer.scheduler``.
        nodes (Nodes): The nodes of the model. The function reads the
            ``name``, ``module``, ``finetuning``, and ``unfreeze_after``
            of each `NodeWrapper`.
        strategy (BaseTrainingStrategy | None): The training strategy,
            or ``None``.

    Returns:
        TrainingPlan: The plan.

    Raises:
        ValueError: When a ``finetuning`` entry claims no parameter, for
            example because earlier entries claimed all its matches.
            Also when a legacy strategy claims a parameter that a
            ``finetuning`` entry claimed.
        KeyError: When an optimizer or a scheduler name is not in its
            registry.

    """
    epochs = cfg.trainer.epochs
    base_optimizer, base_scheduler = _base_configs(cfg, strategy)
    tail = Rule(
        label="default",
        selector=_match_all,
        optimizer=OptimizerSpec.from_config(base_optimizer),
        scheduler=SchedulerSpec.from_config(base_scheduler, epochs),
    )

    builder = _PlanBuilder()
    any_node_rules = any(node.finetuning for node in nodes.values())

    for node in nodes.values():
        _claim_node_rules(
            builder, node, base_optimizer, base_scheduler, epochs
        )
        if strategy is None:
            # Running the tail inside the node loop preserves the group
            # and optimizer creation order of the previous
            # implementation.
            scope = _tail_scope(node, any_node_rules)
            builder.claim(tail, node.name, node.module, scope)

    if strategy is not None:
        _claim_strategy_rules(builder, strategy, nodes, base_scheduler, epochs)
        for node in nodes.values():
            scope = _tail_scope(node, per_node=False)
            builder.claim(tail, node.name, node.module, scope)

    return builder.finish()


@dataclass
class TrainingPlanRuntime:
    """The optimizers and schedulers that `build_training_plan` creates.

    The runtime also gives access to a parameter group through its
    `GroupHandle`. The freeze schedule and the training strategy use
    this access to change the options of their groups, such as the
    learning rate.

    Attributes:
        plan (TrainingPlan): The plan of the runtime.
        inner_optimizers (``tuple[Optimizer, ...]``): One optimizer for
            each `InnerSpec` of the plan, in plan order. The optimizers
            of a legacy training strategy follow them.
        members (``tuple[LRScheduler | ReduceLROnPlateau | None, ...]``):
            The scheduler of each inner optimizer, in the same order.
            The entry is ``None`` for a legacy optimizer without a
            scheduler.
        optimizer (torch.optim.Optimizer): The optimizer that Lightning
            receives. It is the only inner optimizer, or a
            `CompositeOptimizer` over all inner optimizers.
        scheduler_configs (``list[Any]``): The schedulers and scheduler
            configs that Lightning receives. `build_training_plan`
            describes the entries.

    """

    plan: TrainingPlan
    inner_optimizers: tuple[Optimizer, ...]
    members: tuple[LRScheduler | ReduceLROnPlateau | None, ...]
    optimizer: Optimizer
    scheduler_configs: list[Any]

    def group(self, handle: GroupHandle) -> dict[str, Any]:
        """Return the parameter group at a handle.

        Args:
            handle (GroupHandle): The address of the group.

        Returns:
            ``dict[str, Any]``: The entry of ``param_groups`` in the
            inner optimizer. It is not a copy, so a change to it changes
            the optimizer.

        """
        optimizer = self.inner_optimizers[handle.inner_index]
        return optimizer.param_groups[handle.group_index]

    def handles_for_node(self, node_name: str) -> tuple[GroupHandle, ...]:
        """Return the handles of the groups of a node.

        Args:
            node_name (str): The name of the node.

        Returns:
            ``tuple[GroupHandle, ...]``: The handles of the groups that
            hold parameters of the node, in plan order. The tuple is
            empty when no group holds a parameter of ``node_name``.

        """
        return self.plan.handles_by_node.get(node_name, ())

    def set_group_base_lr(self, handle: GroupHandle, lr: float) -> None:
        """Set a new base learning rate for one parameter group.

        The method sets ``lr`` and ``initial_lr`` of the group. It also
        sets the entry of the group in the ``base_lrs`` of the member
        scheduler. For a ``SequentialLR`` or a ``ChainedScheduler``, it
        sets the entry in each child scheduler. The next scheduler steps
        therefore start from the new rate. The method skips a scheduler
        without ``base_lrs``, such as a ``ReduceLROnPlateau``. When the
        inner optimizer has no scheduler, only the group changes.

        Args:
            handle (GroupHandle): The address of the group.
            lr (float): The new base learning rate.

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
    """Create the optimizers and the schedulers of a plan.

    For each inner optimizer of the plan, the function creates the
    optimizer from the ``OPTIMIZERS`` registry. Each group of the plan
    becomes one parameter group with the group options. When the plan
    has more than one group, each parameter group also gets the
    ``name`` of its group. The ``LearningRateMonitor`` key of each group
    then ends with the group name instead of ``pg1``, ``pg2``, and so
    on.

    For each of these optimizers, the function also creates the member
    scheduler from the ``SCHEDULERS`` registry. A ``SequentialLR`` is
    the exception: the function creates it directly from torch. Its
    ``params`` give the ``milestones``, the ``last_epoch``, and the
    child schedulers, which come from the registry. A
    ``ReduceLROnPlateau`` monitors ``main_metric_monitor`` in ``max``
    mode and ``val/loss`` in any other mode.

    The optimizers of `BaseTrainingStrategy.opaque_inners` follow the
    optimizers of the plan. Only a legacy strategy returns such
    optimizers.

    With one optimizer in total, Lightning receives that optimizer and
    its scheduler:

    - A ``ReduceLROnPlateau`` goes into a dictionary with the keys
      ``scheduler``, ``monitor``, and ``frequency``.
    - Another scheduler goes as it is.
    - The scheduler or config of a legacy strategy goes as it is. A
      legacy optimizer without a scheduler gives no scheduler entry.

    With several optimizers, one `CompositeOptimizer` wraps them.
    Lightning then receives these scheduler configs:

    - One `CompositeLRScheduler`, named ``lr``, that steps all
      schedulers without a monitor. It is present only when at least
      one scheduler has no monitor.
    - One `CompositeReduceLROnPlateau` for each monitor, with the
      ``monitor``, ``frequency``, and ``reduce_on_plateau`` keys. The
      configs are named ``lr-plateau``, ``lr-plateau-1``, and so on.

    In this case, the function reads only the ``scheduler`` and the
    ``monitor`` keys of a legacy scheduler config. A legacy scheduler
    with a ``monitor`` must be a ``ReduceLROnPlateau``.

    `CompositeOptimizer` raises ``ValueError`` when one of the several
    optimizers is an ``LBFGS`` optimizer. It also raises ``ValueError``
    when the plan and the strategy give no optimizer.

    Each ``frequency`` above is ``trainer.validation_interval``.

    Args:
        plan (TrainingPlan): The plan from `resolve_training_plan`.
        cfg (Config): The config. The function reads
            ``trainer.validation_interval``.
        main_metric_monitor (str | None): The logged name of the main
            metric, or ``None`` when the model has no main metric.
        strategy (BaseTrainingStrategy | None): The training strategy,
            or ``None``. The function reads its
            `BaseTrainingStrategy.opaque_inners`.

    Returns:
        TrainingPlanRuntime: The optimizers, the schedulers, and the
        scheduler configs for Lightning.

    Raises:
        TypeError: When a group option is not a key of the
            ``defaults`` of its optimizer.
        ValueError: When a ``ReduceLROnPlateau`` in ``max`` mode has no
            ``main_metric_monitor``.

    """
    inner_optimizers: list[Optimizer] = []
    entries: list[_MemberEntry] = []
    # `LearningRateMonitor` suffixes its key with the group name, but
    # only falls back to positional `pg1`, `pg2`, ... when there is
    # more than one group. A lone group is left unnamed so its key
    # stays the bare `lr-<optimizer>` of a plain config.
    name_groups = sum(len(inner.groups) for inner in plan.inners) > 1
    for inner in plan.inners:
        optimizer = _build_inner_optimizer(inner, name_groups)
        inner_optimizers.append(optimizer)
        entries.append(
            _member_entry(inner.scheduler, optimizer, main_metric_monitor)
        )

    bypass_configs = _mount_opaque_inners(strategy, inner_optimizers, entries)

    validation_interval = cfg.trainer.validation_interval

    if len(inner_optimizers) == 1:
        if bypass_configs is None:
            bypass_configs = _bypass_scheduler_configs(
                entries[0], validation_interval
            )
        return TrainingPlanRuntime(
            plan=plan,
            inner_optimizers=tuple(inner_optimizers),
            members=tuple(entry[0] for entry in entries),
            optimizer=inner_optimizers[0],
            scheduler_configs=bypass_configs,
        )

    composite = CompositeOptimizer(inner_optimizers)
    return TrainingPlanRuntime(
        plan=plan,
        inner_optimizers=tuple(inner_optimizers),
        members=tuple(entry[0] for entry in entries),
        optimizer=composite,
        scheduler_configs=_composite_scheduler_configs(
            composite, entries, validation_interval
        ),
    )


def _match_all(
    module: nn.Module,
    module_name: str,
    parameter: nn.Parameter,
    parameter_name: str,
) -> bool:
    _ = module, module_name, parameter, parameter_name
    return True


def _spec_key(name: str, params: Params) -> str:
    return json.dumps(
        {"name": name, "params": params}, sort_keys=True, default=repr
    )


def _unique_name(name: str, used: set[str]) -> str:
    """Reserve a unique name in ``used``.

    Args:
        name (str): The wanted name.
        used (set[str]): The names in use. The function adds the result
            to this set.

    Returns:
        str: ``name`` when ``used`` does not hold it. Otherwise the first
        free name of ``<name>-2``, ``<name>-3``, and so on.

    Example:
        >>> used = {"default"}
        >>> _unique_name("default", used), _unique_name("default", used)
        ('default-2', 'default-3')
        >>> sorted(used)
        ['default', 'default-2', 'default-3']

    """
    unique = name
    index = 2
    while unique in used:
        unique = f"{name}-{index}"
        index += 1
    used.add(unique)
    return unique


def _base_configs(
    cfg: Config, strategy: "BaseTrainingStrategy | None"
) -> tuple[OptimizerConfig, SchedulerConfig]:
    if strategy is None:
        return cfg.trainer.optimizer, cfg.trainer.scheduler
    try:
        return strategy.get_base_configs()
    except NotImplementedError:
        return cfg.trainer.optimizer, cfg.trainer.scheduler


def _tail_scope(node: "NodeWrapper", per_node: bool) -> str:
    # Freezing-scheduled nodes get their own tail group so that
    # `lr_after_unfreeze` has a well-scoped target (the node-purity
    # invariant). `per_node` gives every other node its own group as
    # well, which preserves the previous observable grouping when
    # finetuning rules are present and no strategy runs.
    if node.unfreeze_after is not None or per_node:
        return node.name
    return _SHARED


def _claim_node_rules(
    builder: _PlanBuilder,
    node: "NodeWrapper",
    base_optimizer: OptimizerConfig,
    base_scheduler: SchedulerConfig,
    epochs: int,
) -> None:
    for index, finetuning in enumerate(node.finetuning):
        rule = Rule(
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


def _claim_strategy_rules(
    builder: _PlanBuilder,
    strategy: "BaseTrainingStrategy",
    nodes: "Nodes",
    base_scheduler: SchedulerConfig,
    epochs: int,
) -> None:
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
            scope = node.name if node.unfreeze_after is not None else _SHARED
            builder.claim(rule, node.name, node.module, scope)


def _build_inner_optimizer(inner: InnerSpec, name_groups: bool) -> Optimizer:
    torch_groups = [
        {
            "params": list(group.parameters),
            **group.options,
            **({"name": group.name} if name_groups else {}),
        }
        for group in inner.groups
    ]
    optimizer = from_registry(
        OPTIMIZERS, inner.optimizer_name, params=torch_groups
    )
    optimizer_keys = set(optimizer.defaults) | {"params", "name"}
    for group in optimizer.param_groups:
        unknown_keys = set(group) - optimizer_keys
        if unknown_keys:
            keys = ", ".join(sorted(unknown_keys))
            raise TypeError(
                f"Invalid parameter group option(s) for optimizer "
                f"'{inner.optimizer_name}': {keys}"
            )
    return optimizer


def _member_entry(
    spec: SchedulerSpec,
    optimizer: Optimizer,
    main_metric_monitor: str | None,
) -> _MemberEntry:
    member = _build_member_scheduler(spec, optimizer)
    if isinstance(member, ReduceLROnPlateau):
        return member, _plateau_monitor(spec, main_metric_monitor)
    return member, None


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


def _mount_opaque_inners(
    strategy: "BaseTrainingStrategy | None",
    inner_optimizers: list[Optimizer],
    entries: list[_MemberEntry],
) -> list[_BypassConfig] | None:
    if strategy is None:
        return None
    bypass_configs: list[_BypassConfig] | None = None
    for optimizer, scheduler in strategy.opaque_inners():
        inner_optimizers.append(optimizer)
        if scheduler is None:
            entries.append((None, None))
            continue
        if bypass_configs is None:
            bypass_configs = [scheduler]
        if isinstance(scheduler, dict):
            entries.append((scheduler["scheduler"], scheduler.get("monitor")))
        else:
            entries.append((scheduler, None))
    return bypass_configs


def _bypass_scheduler_configs(
    entry: _MemberEntry, validation_interval: int
) -> list[_BypassConfig]:
    member, monitor = entry
    if member is None:  # pragma: no cover
        return []
    if monitor is not None:
        return [
            {
                "scheduler": member,
                "monitor": monitor,
                "frequency": validation_interval,
            }
        ]
    return [member]


def _composite_scheduler_configs(
    composite: CompositeOptimizer,
    entries: list[_MemberEntry],
    validation_interval: int,
) -> list[LRSchedulerConfigType]:
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

    scheduler_configs: list[LRSchedulerConfigType] = []
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
    return scheduler_configs
