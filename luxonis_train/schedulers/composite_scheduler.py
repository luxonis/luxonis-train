"""The schedulers that step the member schedulers of a
`CompositeOptimizer` together, and a helper that changes a base
learning rate.
"""

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
    """Set the base learning rate of one parameter group in a scheduler.

    The function sets ``base_lrs[index]`` of ``scheduler``. For a
    ``SequentialLR`` or a ``ChainedScheduler``, it sets the entry in
    each child scheduler, and in the children of a child. A scheduler
    without ``base_lrs``, such as a ``ReduceLROnPlateau``, does not
    change. The function does not change the ``lr`` or the
    ``initial_lr`` of the group.
    `TrainingPlanRuntime.set_group_base_lr` sets them.

    Args:
        scheduler (``LRScheduler | ReduceLROnPlateau``): The scheduler.
        index (int): The index of the parameter group in the optimizer
            of the scheduler.
        lr (float): The new base learning rate.

    Example:
        >>> from torch import nn
        >>> from torch.optim import SGD
        >>> from torch.optim.lr_scheduler import StepLR
        >>> optimizer = SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        >>> scheduler = StepLR(optimizer, step_size=1)
        >>> rebase_scheduler_lr(scheduler, 0, 0.5)
        >>> scheduler.base_lrs
        [0.5]
        >>> optimizer.param_groups[0]["lr"]
        0.1

    """
    children = getattr(scheduler, "_schedulers", None)
    if children is not None:
        for child in children:
            rebase_scheduler_lr(child, index, lr)
    base_lrs = getattr(scheduler, "base_lrs", None)
    if base_lrs is not None:
        base_lrs[index] = lr


class CompositeLRScheduler(LRScheduler):
    """One scheduler that steps the member schedulers of a
    `CompositeOptimizer`.

    Each member scheduler belongs to one inner optimizer of the
    composite, so its ``base_lrs`` match the groups of that optimizer.
    Lightning needs ``scheduler.optimizer`` to be the optimizer that
    ``configure_optimizers`` returns. This class gives Lightning one
    scheduler whose ``optimizer`` is the composite.

    The constructor does not call ``LRScheduler.__init__``. The members
    already made their initial step and patched the ``step`` counters
    of their optimizers.

    Example:
        >>> from torch import nn
        >>> from torch.optim import SGD
        >>> from torch.optim.lr_scheduler import StepLR
        >>> first = SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        >>> second = SGD(nn.Linear(2, 2).parameters(), lr=1.0)
        >>> composite = CompositeOptimizer([first, second])
        >>> scheduler = CompositeLRScheduler(
        ...     composite,
        ...     [StepLR(first, 1, gamma=0.5), StepLR(second, 1, gamma=0.1)],
        ... )
        >>> composite.step()
        >>> scheduler.step()
        >>> [round(lr, 3) for lr in scheduler.get_last_lr()]
        [0.05, 0.1]

    """

    def __init__(
        self,
        composite: CompositeOptimizer,
        members: Sequence[LRScheduler],
    ):
        """Wrap the member schedulers.

        The scheduler sets ``last_epoch`` to ``0``.

        Args:
            composite (CompositeOptimizer): The optimizer that Lightning
                receives. The scheduler stores it as ``optimizer``.
            members (``Sequence[LRScheduler]``): The member schedulers.
                Each one belongs to an inner optimizer of ``composite``.

        """
        self.optimizer = composite
        self._members = tuple(members)
        self.last_epoch = 0

    @property
    def members(self) -> tuple[LRScheduler, ...]:
        """The member schedulers, in the order of the constructor."""
        return self._members

    def step(self, epoch: int | None = None) -> None:  # type: ignore[override]
        """Step every member scheduler once.

        The method adds ``1`` to ``last_epoch``, and then calls
        ``step()`` of each member in order.

        Args:
            epoch (int | None): The epoch that Lightning can pass. The
                method ignores it.

        """
        _ = epoch
        self.last_epoch += 1
        for member in self._members:
            member.step()

    def get_last_lr(self) -> "list[float | Tensor]":
        """Return the last learning rates of all member schedulers.

        Returns:
            ``list[float | Tensor]``: The ``get_last_lr()`` values of the
            members, joined in member order. The list has one value for
            each group of each member optimizer.

        """
        return [lr for member in self._members for lr in member.get_last_lr()]

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler and its members.

        Returns:
            ``dict[str, Any]``: A dictionary with these keys:

            - ``"version"``: ``1``.
            - ``"last_epoch"``: The ``last_epoch`` counter.
            - ``"members"``: The ``state_dict()`` of each member.

        """
        return {
            "version": 1,
            "last_epoch": self.last_epoch,
            "members": [member.state_dict() for member in self._members],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a state into the scheduler and its members.

        The method restores ``last_epoch`` and loads each entry of
        ``"members"`` into the member at the same position.

        Args:
            state_dict (``dict[str, Any]``): A state that the
                `CompositeLRScheduler.state_dict` method returned.

        Raises:
            ValueError: If ``"version"`` is not ``1``. Also if the
                number of member states differs from the number of
                members.

        """
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
    """One ``ReduceLROnPlateau`` that steps the plateau member
    schedulers of a `CompositeOptimizer`.

    The class does the job of `CompositeLRScheduler` for members that
    monitor a value. Each member is a ``ReduceLROnPlateau`` over one
    inner optimizer of the composite. Lightning passes the monitored
    value as the first argument of `step`.

    The constructor does not call ``ReduceLROnPlateau.__init__``. The
    members hold the plateau settings, such as ``mode`` and
    ``patience``.

    """

    def __init__(
        self,
        composite: CompositeOptimizer,
        members: Sequence[ReduceLROnPlateau],
    ):
        """Wrap the member schedulers.

        Args:
            composite (CompositeOptimizer): The optimizer that Lightning
                receives. The scheduler stores it as ``optimizer``.
            members (``Sequence[ReduceLROnPlateau]``): The member
                schedulers. Each one belongs to an inner optimizer of
                ``composite``.

        """
        self.optimizer = composite
        self._members = tuple(members)

    @property
    def members(self) -> tuple[ReduceLROnPlateau, ...]:
        """The member schedulers, in the order of the constructor."""
        return self._members

    def step(  # type: ignore[override]
        self, metrics: Any, epoch: int | None = None
    ) -> None:
        """Pass the monitored value to every member scheduler.

        Each member decides on its own whether to reduce the learning
        rate of its optimizer.

        Args:
            metrics (``Any``): The monitored value, such as the
                validation loss.
            epoch (int | None): The epoch that Lightning can pass. The
                method ignores it.

        """
        _ = epoch
        for member in self._members:
            member.step(metrics)

    def state_dict(self) -> dict[str, Any]:
        """Return the states of the member schedulers.

        Returns:
            ``dict[str, Any]``: A dictionary with the keys ``"version"``,
            which is ``1``, and ``"members"``, which holds the
            ``state_dict()`` of each member.

        """
        return {
            "version": 1,
            "members": [member.state_dict() for member in self._members],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a state into the member schedulers.

        The method loads each entry of ``"members"`` into the member at
        the same position.

        Args:
            state_dict (``dict[str, Any]``): A state that the
                `CompositeReduceLROnPlateau.state_dict` method returned.

        Raises:
            ValueError: If ``"version"`` is not ``1``. Also if the
                number of member states differs from the number of
                members.

        """
        if state_dict.get("version") != 1:
            raise ValueError(
                "Unsupported composite scheduler checkpoint version: "
                f"{state_dict.get('version')!r}."
            )
        for member, member_state in zip(
            self._members, state_dict["members"], strict=True
        ):
            member.load_state_dict(member_state)
