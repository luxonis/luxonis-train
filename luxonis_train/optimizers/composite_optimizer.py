"""An optimizer that drives several inner optimizers.

Lightning sees one optimizer, so gradient accumulation and gradient
clipping still work for any number of inner optimizers that the
finetuning rules and the training strategy produce.

"""

from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.optim import LBFGS, Optimizer

__all__ = ["CompositeOptimizer", "unwrap_optimizers"]


def _intersect_defaults(inners: Sequence[Optimizer]) -> dict[str, Any]:
    """Return the ``defaults`` keys that all inner optimizers share.

    ``LearningRateMonitor`` reads ``param_group["betas"][0]`` of every
    group when ``"betas"`` is in ``optimizer.defaults``. A key therefore
    stays only when every inner optimizer, and so every parameter group,
    has it.

    Args:
        inners (``Sequence[Optimizer]``): The inner optimizers. The
            sequence must not be empty.

    Returns:
        ``dict[str, Any]``: The shared keys, with the values of the first
        inner optimizer.

    """
    keys = set(inners[0].defaults)
    for inner in inners[1:]:
        keys &= set(inner.defaults)
    return {key: inners[0].defaults[key] for key in keys}


class _CompositeState(MutableMapping[Tensor, Any]):
    """A live view over the states of the inner optimizers.

    A read goes through the ``state`` mappings of the inner optimizers
    in order. A write goes to the inner optimizer that owns the
    parameter. Code that assigns ``optimizer.state[p]``, such as
    ``_optimizer_to_device`` of Lightning, therefore works with the
    composite. A key that no inner optimizer owns raises ``KeyError``.

    """

    def __init__(self, inners: Sequence[Optimizer]):
        self._inners = tuple(inners)
        self._owner = {
            id(parameter): inner
            for inner in inners
            for group in inner.param_groups
            for parameter in group["params"]
        }

    def _owner_of(self, key: Tensor) -> Optimizer:
        owner = self._owner.get(id(key))
        if owner is None:
            raise KeyError(key)
        return owner

    def __getitem__(self, key: Tensor) -> Any:
        return self._owner_of(key).state[key]

    def __setitem__(self, key: Tensor, value: Any) -> None:
        self._owner_of(key).state[key] = value

    def __delitem__(self, key: Tensor) -> None:
        del self._owner_of(key).state[key]

    def __iter__(self) -> Iterator[Tensor]:
        for inner in self._inners:
            yield from inner.state

    def __len__(self) -> int:
        return sum(len(inner.state) for inner in self._inners)


class CompositeOptimizer(Optimizer):
    """One `torch.optim.Optimizer` that drives several inner optimizers.

    `param_groups` joins the ``param_groups`` of the inner optimizers,
    and holds the same dictionary objects. Lightning can therefore drive
    several optimizer configurations in its automatic optimization:

    - one ``step`` call,
    - one gradient clipping pass over all groups,
    - one gradient scaler slot.

    The partition of the parameters is static. No group joins, leaves,
    or moves after the constructor. When a node freezes, only
    ``requires_grad`` changes, and an inner optimizer skips a parameter
    whose gradient is ``None``.

    The constructor does not call ``Optimizer.__init__``, because that
    method builds its own parameter groups. The class itself implements
    the methods that Lightning uses: the ``Optimizable`` protocol,
    `step`, `zero_grad`, `state_dict`, and `load_state_dict`.

    Example:
        >>> from torch import nn
        >>> from torch.optim import SGD, Adam
        >>> sgd = SGD(nn.Linear(4, 8).parameters(), lr=0.1)
        >>> adam = Adam(nn.Linear(8, 2).parameters(), lr=0.01)
        >>> composite = CompositeOptimizer([sgd, adam])
        >>> [group["lr"] for group in composite.param_groups]
        [0.1, 0.01]
        >>> "betas" in composite.defaults
        False
        >>> composite.state_dict()["optimizers"]
        ['SGD', 'Adam']

    """

    STATE_DICT_FORMAT = "luxonis_composite"

    def __init__(self, inners: Sequence[Optimizer]):
        """Wrap the inner optimizers.

        The ``defaults`` of the composite hold only the keys that all
        inner optimizers share, with the values of the first one.

        Args:
            inners (``Sequence[Optimizer]``): The inner optimizers, in
                the order of their groups in `param_groups`.

        Raises:
            ValueError: If ``inners`` is empty. Also if ``inners`` has
                more than one optimizer and one of them is an ``LBFGS``
                optimizer. ``LBFGS`` needs the step closure, and `step`
                does not pass the closure to the inner optimizers.

        """
        if not inners:
            raise ValueError(
                "`CompositeOptimizer` requires at least one optimizer."
            )
        if len(inners) > 1 and any(
            isinstance(inner, LBFGS) for inner in inners
        ):
            raise ValueError(
                "Optimizers that require a step closure ('LBFGS') cannot "
                "be combined with other optimizers. Use a single "
                "optimizer/scheduler configuration instead."
            )
        # NOTE: No attribute may be called `optimizer` -
        # `LearningRateMonitor` unwraps that exact name.
        self._inners = tuple(inners)
        self._state_view = _CompositeState(self._inners)
        self.defaults = _intersect_defaults(self._inners)

        # torch-compat: inherited helpers (hook registration,
        # `__setstate__`) expect these to exist. The composite's `step`
        # is intentionally not wrapped by `_patch_step_function` - the
        # inner steps already fire the global and per-instance torch
        # hooks, and wrapping the facade would double-fire them.
        self._optimizer_step_pre_hooks: OrderedDict = OrderedDict()
        self._optimizer_step_post_hooks: OrderedDict = OrderedDict()
        self._optimizer_state_dict_pre_hooks: OrderedDict = OrderedDict()
        self._optimizer_state_dict_post_hooks: OrderedDict = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks: OrderedDict = OrderedDict()
        self._optimizer_load_state_dict_post_hooks: OrderedDict = OrderedDict()

    @property
    def inner_optimizers(self) -> tuple[Optimizer, ...]:
        """The inner optimizers, in the order of the constructor."""
        return self._inners

    @property  # type: ignore[override]
    def param_groups(self) -> list[dict[str, Any]]:
        """The parameter groups of all inner optimizers, in order.

        Each access builds a new list, but the list holds the group
        dictionaries of the inner optimizers. A change to a group
        therefore changes its inner optimizer. An assignment to the
        property raises ``TypeError``.

        """
        # Recomputed on access on purpose: `Optimizer.load_state_dict`
        # replaces the inner group dictionaries, so a stored
        # concatenation would go stale after a checkpoint restore.
        return [
            group for inner in self._inners for group in inner.param_groups
        ]

    @param_groups.setter
    def param_groups(self, value: Any) -> None:
        """Reject a new value, because the partition is fixed."""
        raise TypeError(
            "`CompositeOptimizer.param_groups` cannot be replaced; "
            "the parameter partition is fixed at construction."
        )

    @property  # type: ignore[override]
    def state(self) -> _CompositeState:
        """A live view over the states of the inner optimizers.

        A write to ``state[parameter]`` goes to the inner optimizer that
        owns the parameter. A parameter that no inner optimizer owns
        raises ``KeyError``. An assignment to the property raises
        ``TypeError``.

        """
        return self._state_view

    @state.setter
    def state(self, value: Any) -> None:
        """Reject a new value, because the state is a view."""
        raise TypeError(
            "`CompositeOptimizer.state` cannot be replaced; it is a "
            "view over the inner optimizers' states."
        )

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Callable[[], Any] | None = None
    ) -> Any:
        """Run the closure once, then step every inner optimizer.

        The method calls ``closure`` with gradients on. It then calls
        ``step()`` of each inner optimizer in order, without a closure.
        The inner steps fire the step hooks of torch. The composite
        fires no hooks of its own.

        Args:
            closure (``Callable[[], Any] | None``): The function that
                computes the loss and the gradients, or ``None``.

        Returns:
            ``Any``: The return value of ``closure``, or ``None`` without
            a closure.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for inner in self._inners:
            inner.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the gradients of every inner optimizer.

        Args:
            set_to_none (bool): Whether to set the gradients to ``None``
                instead of to zero. The method passes it to each inner
                optimizer.

        """
        for inner in self._inners:
            inner.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Reject a new parameter group, because the partition is fixed.

        Args:
            param_group (``dict[str, Any]``): The group. The method does
                not use it.

        Raises:
            RuntimeError: Always.

        """
        raise RuntimeError(
            "`CompositeOptimizer` is a fixed partition of the model "
            "parameters; groups cannot be added after construction."
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the state of every inner optimizer.

        Returns:
            ``dict[str, Any]``: A dictionary with these keys:

            - ``"format"``: ``"luxonis_composite"``.
            - ``"version"``: ``1``.
            - ``"optimizers"``: The class name of each inner optimizer.
            - ``"inners"``: The ``state_dict()`` of each inner optimizer.

        """
        return {
            "format": self.STATE_DICT_FORMAT,
            "version": 1,
            "optimizers": [type(inner).__name__ for inner in self._inners],
            "inners": [inner.state_dict() for inner in self._inners],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a composite state into the inner optimizers.

        The method loads each entry of ``"inners"`` into the inner
        optimizer at the same position.

        Args:
            state_dict (``dict[str, Any]``): A state that the
                `CompositeOptimizer.state_dict` method returned.

        Raises:
            ValueError: If ``"format"`` is not ``"luxonis_composite"``,
                for example in the checkpoint of a single optimizer. Also
                if ``"version"`` is not ``1``, or if the class names in
                ``"optimizers"`` differ from the current inner
                optimizers.

        """
        if state_dict.get("format") != self.STATE_DICT_FORMAT:
            raise ValueError(
                "The checkpoint was saved with a different optimizer "
                "configuration (a single optimizer or a pre-release "
                "multi-optimizer build) and cannot be loaded into a "
                "`CompositeOptimizer`."
            )
        if state_dict.get("version") != 1:
            raise ValueError(
                "Unsupported `CompositeOptimizer` checkpoint version: "
                f"{state_dict.get('version')!r}."
            )
        expected = [type(inner).__name__ for inner in self._inners]
        found = state_dict.get("optimizers")
        if found != expected:
            raise ValueError(
                "The checkpointed optimizer configuration does not "
                f"match the current one. Checkpoint: {found}, "
                f"current: {expected}."
            )
        for inner, inner_state in zip(
            self._inners, state_dict["inners"], strict=True
        ):
            inner.load_state_dict(inner_state)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        inners = "\n".join(
            f"  #{index}: " + repr(inner).replace("\n", "\n  ")
            for index, inner in enumerate(self._inners)
        )
        return f"{type(self).__name__} (\n{inners}\n)"


def unwrap_optimizers(
    optimizers: Sequence[Optimizer],
) -> list[Optimizer]:
    """Replace each `CompositeOptimizer` with its inner optimizers.

    A plain optimizer stays as it is. A caller can therefore treat a
    run with one optimizer and a run with a composite in the same way.

    Args:
        optimizers (``Sequence[Optimizer]``): The optimizers, such as
            ``trainer.optimizers`` of Lightning.

    Returns:
        ``list[Optimizer]``: A new list with the plain optimizers, in
        order.

    Example:
        >>> from torch import nn
        >>> from torch.optim import SGD, Adam
        >>> sgd = SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        >>> adam = Adam(nn.Linear(2, 2).parameters(), lr=0.01)
        >>> composite = CompositeOptimizer([sgd, adam])
        >>> unwrap_optimizers([composite]) == [sgd, adam]
        True
        >>> unwrap_optimizers([sgd]) == [sgd]
        True

    """
    unwrapped: list[Optimizer] = []
    for optimizer in optimizers:
        if isinstance(optimizer, CompositeOptimizer):
            unwrapped.extend(optimizer.inner_optimizers)
        else:
            unwrapped.append(optimizer)
    return unwrapped
