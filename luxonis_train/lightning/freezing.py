"""The freeze schedule of the nodes, derived from the config.

The config alone defines the schedule. A node with ``freezing.active``
stays frozen for each epoch before its unfreeze epoch, and trains from
that epoch on. `resolve_unfreeze_epoch` computes the unfreeze epoch.

`FreezeSchedule.apply` derives the state of the nodes from the epoch
number, so a repeated call for the same epoch gives the same state.
`TrainingManager` calls it when a fit starts and at the start of each
training epoch. A resumed run thus gets the correct state, and the
checkpoint needs no state of the callback:

- A frozen parameter stays in its parameter group, because the
  training plan puts each parameter in one fixed group. The optimizer
  state thus loads as a plain ``state_dict``.
- Each epoch start sets ``requires_grad`` and the batch normalization
  state again from the schedule.
- The optimizer and scheduler state dicts hold the learning rates.
  Lightning saves and restores them.

`FreezeSchedule.apply` sets ``lr_after_unfreeze`` only on the unfreeze
epoch itself. A run that resumes *after* that epoch does not set it
again. The checkpoint already holds the learning rate that the
scheduler reached. A second update of the rate would break a scheduler
that computes each rate from the previous rate, such as ``StepLR``.

"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from typing_extensions import Self

from luxonis_train.config.config import FreezingConfig

if TYPE_CHECKING:
    from luxonis_train.lightning.training_plan import (
        GroupHandle,
        TrainingPlanRuntime,
    )
    from luxonis_train.lightning.utils import Nodes

__all__ = ["FreezeSchedule", "NodeFreezePlan", "resolve_unfreeze_epoch"]


def resolve_unfreeze_epoch(
    freezing: FreezingConfig, total_epochs: int
) -> int | None:
    """Resolve ``freezing.unfreeze_after`` to an epoch number.

    An integer is the epoch number itself. A float is a share of
    ``total_epochs``, truncated to an integer. ``None`` gives
    ``total_epochs``, so the node stays frozen for the whole run.

    Args:
        freezing (FreezingConfig): The ``freezing`` section of a node
            config.
        total_epochs (int): The number of epochs of the run, from
            ``trainer.epochs``.

    Returns:
        int | None: The first epoch in which the node trains. ``None``
        when ``freezing.active`` is ``False``.

    Example:
        >>> from luxonis_train.config.config import FreezingConfig
        >>> resolve_unfreeze_epoch(
        ...     FreezingConfig(active=True, unfreeze_after=0.25), 10
        ... )
        2
        >>> resolve_unfreeze_epoch(FreezingConfig(active=True), 10)
        10
        >>> print(resolve_unfreeze_epoch(FreezingConfig(), 10))
        None

    """
    if not freezing.active:
        return None
    if freezing.unfreeze_after is None:
        return total_epochs
    if isinstance(freezing.unfreeze_after, int):
        return freezing.unfreeze_after
    return int(freezing.unfreeze_after * total_epochs)


@dataclass
class NodeFreezePlan:
    """The freeze schedule of one node, with its original state.

    Attributes:
        node_name (str): The name of the node in the graph.
        unfreeze_epoch (int): The first epoch in which the node trains.
        lr_after_unfreeze (float | None): The base learning rate of the
            parameter groups of the node from the unfreeze epoch on.
            ``None`` keeps the rate that the scheduler reached.
        parameters (``list[nn.Parameter]``): The parameters of the node.
        original_requires_grad (list[bool]): The ``requires_grad``
            value of each parameter before any freeze, in the order of
            ``parameters``.
        batch_norms (``list[_BatchNorm]``): The batch normalization
            layers of the node.
        original_track_running_stats (list[bool]): The
            ``track_running_stats`` value of each layer before any
            freeze, in the order of ``batch_norms``.
        group_handles (``tuple[GroupHandle, ...]``): The parameter
            groups that hold the parameters of the node.
            `FreezeSchedule.attach_group_handles` sets them. The tuple
            is empty before that call.

    """

    node_name: str
    unfreeze_epoch: int
    lr_after_unfreeze: float | None
    parameters: list[nn.Parameter]
    original_requires_grad: list[bool]
    batch_norms: list[_BatchNorm]
    original_track_running_stats: list[bool]
    group_handles: tuple["GroupHandle", ...] = field(default_factory=tuple)

    @classmethod
    def from_module(
        cls,
        node_name: str,
        module: nn.Module,
        unfreeze_epoch: int,
        lr_after_unfreeze: float | None,
    ) -> Self:
        """Create a plan that records the original state of a module.

        The plan stores the ``requires_grad`` value of each parameter
        and the ``track_running_stats`` value of each batch
        normalization layer. An unfreeze restores these values. It does
        not make every parameter trainable. A parameter that the node
        itself freezes, or a layer built with
        ``track_running_stats=False``, thus keeps its setting. Call this
        method before any freeze.

        Args:
            node_name (str): The name of the node in the graph.
            module (``nn.Module``): The node.
            unfreeze_epoch (int): The first epoch in which the node
                trains.
            lr_after_unfreeze (float | None): The base learning rate of
                the node from the unfreeze epoch on. ``None`` keeps the
                rate that the scheduler reached.

        Returns:
            ``Self``: The plan, without group handles.

        Example:
            >>> from torch import nn
            >>> module = nn.Linear(2, 2)
            >>> _ = module.bias.requires_grad_(False)
            >>> plan = NodeFreezePlan.from_module(
            ...     "head", module, unfreeze_epoch=3, lr_after_unfreeze=None
            ... )
            >>> plan.original_requires_grad
            [True, False]
            >>> plan.is_frozen(2), plan.unfreezes_at(3)
            (True, True)

        """
        parameters = list(module.parameters())
        batch_norms = [
            submodule
            for submodule in module.modules()
            if isinstance(submodule, _BatchNorm)
        ]
        return cls(
            node_name=node_name,
            unfreeze_epoch=unfreeze_epoch,
            lr_after_unfreeze=lr_after_unfreeze,
            parameters=parameters,
            original_requires_grad=[
                parameter.requires_grad for parameter in parameters
            ],
            batch_norms=batch_norms,
            original_track_running_stats=[
                batch_norm.track_running_stats for batch_norm in batch_norms
            ],
        )

    def is_frozen(self, epoch: int) -> bool:
        """Return whether the node is frozen in an epoch.

        Args:
            epoch (int): The epoch number, from ``0``.

        Returns:
            bool: ``True`` when ``epoch`` comes before
            ``unfreeze_epoch``.

        """
        return epoch < self.unfreeze_epoch

    def unfreezes_at(self, epoch: int) -> bool:
        """Return whether the node unfreezes in an epoch.

        Args:
            epoch (int): The epoch number, from ``0``.

        Returns:
            bool: ``True`` when ``epoch`` is ``unfreeze_epoch``.

        """
        return epoch == self.unfreeze_epoch


class FreezeSchedule:
    """The freeze plans of all nodes with ``freezing.active``.

    `Nodes` builds the schedule after it builds the nodes, and stores it
    in ``freeze_schedule``. `TrainingManager` applies it.

    """

    def __init__(self, plans: list[NodeFreezePlan]):
        """Initialize the schedule.

        Args:
            plans (list[NodeFreezePlan]): One plan for each node with a
                freeze schedule. The schedule keeps the list itself, not
                a copy.

        """
        self._plans = plans

    @classmethod
    def from_nodes(cls, nodes: "Nodes") -> Self:
        """Build the schedule from the nodes of a model.

        The method creates a plan with `NodeFreezePlan.from_module` for
        each node whose ``unfreeze_after`` is not ``None``. Build the
        schedule before any freeze, so that the plans record the
        original state.

        Args:
            nodes (Nodes): The nodes of the model. The method reads the
                ``name``, ``module``, ``unfreeze_after``, and
                ``lr_after_unfreeze`` of each `NodeWrapper`.

        Returns:
            ``Self``: The schedule, with the plans in the order of
            ``nodes``.

        """
        return cls(
            [
                NodeFreezePlan.from_module(
                    node_name=node.name,
                    module=node.module,
                    unfreeze_epoch=node.unfreeze_after,
                    lr_after_unfreeze=node.lr_after_unfreeze,
                )
                for node in nodes.values()
                if node.unfreeze_after is not None
            ]
        )

    def __bool__(self) -> bool:
        """Return whether the schedule holds at least one plan.

        Returns:
            bool: ``False`` when no node has a freeze schedule.

        """
        return bool(self._plans)

    @property
    def plans(self) -> list[NodeFreezePlan]:
        """The plans of the schedule, one for each scheduled node.

        The property returns the stored list, not a copy.

        """
        return self._plans

    def is_frozen(self, node_name: str, epoch: int) -> bool:
        """Return whether a node is frozen in an epoch.

        Args:
            node_name (str): The name of the node in the graph.
            epoch (int): The epoch number, from ``0``.

        Returns:
            bool: ``True`` when the node has a plan and ``epoch`` comes
            before its unfreeze epoch. ``False`` for a node without a
            plan.

        """
        for plan in self._plans:
            if plan.node_name == node_name:
                return plan.is_frozen(epoch)
        return False

    def attach_group_handles(self, runtime: "TrainingPlanRuntime") -> None:
        """Store in each plan the parameter groups of its node.

        `LuxonisLightningModule.configure_optimizers` calls the method
        after it builds the optimizers. For each plan, the method takes
        the handles of the groups that hold parameters of the node. It
        checks that each of these groups holds parameters of that node
        only. ``lr_after_unfreeze`` thus changes no learning rate of
        another node.

        Args:
            runtime (TrainingPlanRuntime): The optimizers and schedulers
                of the run.

        Raises:
            RuntimeError: When a group of a scheduled node also holds
                parameters of another node.

        """
        for plan in self._plans:
            handles = runtime.handles_for_node(plan.node_name)
            for handle in handles:
                group_spec = runtime.plan.inners[handle.inner_index].groups[
                    handle.group_index
                ]
                if group_spec.node_names != (
                    plan.node_name,
                ):  # pragma: no cover
                    raise RuntimeError(
                        "Internal error: parameter group "
                        f"'{group_spec.name}' mixes parameters of the "
                        f"frozen node '{plan.node_name}' with other "
                        "nodes."
                    )
            plan.group_handles = handles

    def apply(
        self,
        epoch: int,
        runtime: "TrainingPlanRuntime | None" = None,
    ) -> None:
        """Set the state of each scheduled node for an epoch.

        For a node that is frozen in ``epoch``, the method sets
        ``requires_grad`` of each parameter and ``track_running_stats``
        of each batch normalization layer to ``False``. For any other
        node, it restores the original values of the plan. The method
        logs an info message for a node when a parameter changes from
        trainable to frozen, and another when a parameter changes back.
        A repeated call for the same epoch gives the same state.

        With ``runtime``, the method also sets ``lr_after_unfreeze`` as
        the base learning rate of each group of a node, but only when
        ``epoch`` is the unfreeze epoch of the node. The module
        docstring tells why a later epoch does not set the rate. The
        groups come from `attach_group_handles`.

        Args:
            epoch (int): The epoch number, from ``0``.
            runtime (``TrainingPlanRuntime | None``): The optimizers and
                schedulers of the run. ``None`` changes no learning
                rate.

        Example:
            The example turns the logger off, so the info messages do
            not show:

            >>> from loguru import logger
            >>> from torch import nn
            >>> logger.disable("luxonis_train")
            >>> module = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
            >>> plan = NodeFreezePlan.from_module(
            ...     "head", module, unfreeze_epoch=2, lr_after_unfreeze=None
            ... )
            >>> schedule = FreezeSchedule([plan])
            >>> schedule.apply(epoch=0)
            >>> module[0].weight.requires_grad, module[1].track_running_stats
            (False, False)
            >>> schedule.apply(epoch=2)
            >>> module[0].weight.requires_grad, module[1].track_running_stats
            (True, True)
            >>> logger.enable("luxonis_train")

        """
        for plan in self._plans:
            _apply_plan(plan, epoch, runtime)


def _apply_plan(
    plan: NodeFreezePlan,
    epoch: int,
    runtime: "TrainingPlanRuntime | None",
) -> None:
    frozen = plan.is_frozen(epoch)
    froze, unfroze = _converge_requires_grad(plan, frozen)
    for batch_norm, original in zip(
        plan.batch_norms, plan.original_track_running_stats, strict=True
    ):
        batch_norm.track_running_stats = False if frozen else original
    if froze:
        logger.info(f"Freezing node '{plan.node_name}'")
    if unfroze:
        logger.info(f"Unfreezing node '{plan.node_name}'")
    if (
        runtime is not None
        and plan.unfreezes_at(epoch)
        and plan.lr_after_unfreeze is not None
    ):
        for handle in plan.group_handles:
            runtime.set_group_base_lr(handle, plan.lr_after_unfreeze)


def _converge_requires_grad(
    plan: NodeFreezePlan, frozen: bool
) -> tuple[bool, bool]:
    froze = unfroze = False
    for parameter, original in zip(
        plan.parameters, plan.original_requires_grad, strict=True
    ):
        target = original and not frozen
        if parameter.requires_grad and not target:
            froze = True
        elif target and not parameter.requires_grad:
            unfroze = True
        parameter.requires_grad = target
    return froze, unfroze
