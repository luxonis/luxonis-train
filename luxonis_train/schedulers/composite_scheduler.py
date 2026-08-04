from collections.abc import Sequence
from typing import Any

from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from luxonis_train.optimizers.composite_optimizer import CompositeOptimizer

__all__ = [
    "CompositeLRScheduler",
    "CompositeReduceLROnPlateau",
    "rebase_scheduler_lr",
]


def rebase_scheduler_lr(
    scheduler: LRScheduler | ReduceLROnPlateau, index: int, lr: float
) -> None:
    """Rewrite the base learning rate of one parameter group inside a
    scheduler, recursing into C{SequentialLR}/C{ChainedScheduler}
    children.

    C{ReduceLROnPlateau} has no C{base_lrs}; for it the group-level
    C{lr}/C{initial_lr} writes (done by the caller) are the rebase.
    """
    children = getattr(scheduler, "_schedulers", None)
    if children is not None:
        for child in children:
            rebase_scheduler_lr(child, index, lr)
    base_lrs = getattr(scheduler, "base_lrs", None)
    if base_lrs is not None:
        base_lrs[index] = lr


class CompositeLRScheduler(LRScheduler):
    """Fans one Lightning-facing scheduler out to member schedulers,
    each of which owns one inner optimizer of a L{CompositeOptimizer}.

    Lightning requires C{scheduler.optimizer} to be identical to an
    optimizer returned from C{configure_optimizers}, while the member
    schedulers must be constructed against the inner optimizers (so
    their C{base_lrs} line up with the inner parameter groups) - hence
    this wrapper.

    Deliberately does not call C{LRScheduler.__init__}: the members
    already performed their initial step and patched their own
    optimizers' C{step} counters.
    """

    def __init__(
        self,
        composite: CompositeOptimizer,
        members: Sequence[LRScheduler],
    ):
        self.optimizer = composite
        self._members = tuple(members)
        self.last_epoch = 0

    @property
    def members(self) -> tuple[LRScheduler, ...]:
        return self._members

    def step(self, epoch: int | None = None) -> None:  # type: ignore[override]
        _ = epoch
        self.last_epoch += 1
        for member in self._members:
            member.step()

    def get_last_lr(self) -> "list[float | Tensor]":
        return [lr for member in self._members for lr in member.get_last_lr()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "last_epoch": self.last_epoch,
            "members": [member.state_dict() for member in self._members],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("version") != 1:
            raise ValueError(
                "Unsupported composite scheduler checkpoint version: "
                f"{state_dict.get('version')!r}."
            )
        self.last_epoch = state_dict["last_epoch"]
        for member, member_state in zip(
            self._members, state_dict["members"], strict=True
        ):
            member.load_state_dict(member_state)


class CompositeReduceLROnPlateau(ReduceLROnPlateau):
    """Plateau counterpart of L{CompositeLRScheduler}.

    Members are real C{ReduceLROnPlateau} instances over their inner
    optimizers; Lightning passes the monitored value positionally.
    """

    def __init__(
        self,
        composite: CompositeOptimizer,
        members: Sequence[ReduceLROnPlateau],
    ):
        self.optimizer = composite
        self._members = tuple(members)

    @property
    def members(self) -> tuple[ReduceLROnPlateau, ...]:
        return self._members

    def step(  # type: ignore[override]
        self, metrics: Any, epoch: int | None = None
    ) -> None:
        _ = epoch
        for member in self._members:
            member.step(metrics)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "members": [member.state_dict() for member in self._members],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("version") != 1:
            raise ValueError(
                "Unsupported composite scheduler checkpoint version: "
                f"{state_dict.get('version')!r}."
            )
        for member, member_state in zip(
            self._members, state_dict["members"], strict=True
        ):
            member.load_state_dict(member_state)
