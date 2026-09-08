"""Declarative node freezing.

The freeze/unfreeze schedule is a pure function of the config: a node
with ``freezing.active`` is frozen for every epoch before its resolved
unfreeze epoch and trainable afterwards. L{FreezeSchedule.apply} is
idempotent and is driven at ``setup`` time and at the start of every
training epoch, so a resumed run converges to the correct state without
any checkpointed callback state:

    - parameter group membership never changes (the training plan is a
      total static partition), so optimizer checkpoints are a plain
      ``state_dict`` round trip;
    - the ``requires_grad``/BatchNorm state is re-derived from the
      schedule at every epoch start;
    - learning rates live in the optimizer and scheduler state dicts,
      which Lightning checkpoints and restores natively.

``lr_after_unfreeze`` is applied only on the schedule edge (``epoch ==
unfreeze_epoch``). A run resumed *past* the edge must not re-apply it:
the checkpoint already carries the scheduler-evolved learning rate, and
re-applying would permanently corrupt recursive schedulers such as
``StepLR``.
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

    C{None} means "frozen for the whole run" and resolves to the total
    number of epochs; a float is a fraction of the total.
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
        """Snapshot the module's original trainability state.

        Must be called before any freeze is applied: unfreezing
        restores these snapshots rather than forcing everything
        trainable, so design-frozen parameters and BatchNorm layers
        constructed with ``track_running_stats=False`` keep their
        configuration.
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
        return epoch < self.unfreeze_epoch

    def unfreezes_at(self, epoch: int) -> bool:
        return epoch == self.unfreeze_epoch


class FreezeSchedule:
    """The freeze schedules of all ``freezing.active`` nodes."""

    def __init__(self, plans: list[NodeFreezePlan]):
        self._plans = plans

    @classmethod
    def from_nodes(cls, nodes: "Nodes") -> Self:
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
        return bool(self._plans)

    @property
    def plans(self) -> list[NodeFreezePlan]:
        return self._plans

    def is_frozen(self, node_name: str, epoch: int) -> bool:
        for plan in self._plans:
            if plan.node_name == node_name:
                return plan.is_frozen(epoch)
        return False

    def attach_group_handles(self, runtime: "TrainingPlanRuntime") -> None:
        """Bind each scheduled node to the parameter groups holding its
        parameters, asserting the node-purity invariant that makes
        ``lr_after_unfreeze`` well-scoped.
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
        """Converge the model to the scheduled state for C{epoch}.

        Idempotent: re-derives ``requires_grad`` and BatchNorm
        statistics tracking from the schedule and the original
        snapshots, logging only on actual transitions.
        ``lr_after_unfreeze`` is applied only on the exact unfreeze
        epoch (see the module docstring for why a resumed run past the
        edge must not re-apply it).
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
