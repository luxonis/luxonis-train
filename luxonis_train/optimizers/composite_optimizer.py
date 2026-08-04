from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.optim import LBFGS, Optimizer

__all__ = ["CompositeOptimizer", "unwrap_optimizers"]


def _intersect_defaults(inners: Sequence[Optimizer]) -> dict[str, Any]:
    """Key-intersection of the inner optimizers' defaults.

    C{LearningRateMonitor} indexes C{param_group["betas"][0]} for every
    group whenever C{"betas" in optimizer.defaults}, so a key may only
    survive when every inner optimizer (and therefore every parameter
    group) supports it. Values are taken from the first inner.
    """
    keys = set(inners[0].defaults)
    for inner in inners[1:]:
        keys &= set(inner.defaults)
    return {key: inners[0].defaults[key] for key in keys}


class _CompositeState(MutableMapping[Tensor, Any]):
    """Live view over the inner optimizers' states.

    Reads chain the inner C{state} mappings; writes are routed to the
    inner optimizer owning the parameter, so code like Lightning's
    C{_optimizer_to_device} (which reassigns C{optimizer.state[p]})
    keeps working against the composite.
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
    """A single C{torch.optim.Optimizer} facade over several inner
    optimizers.

    C{param_groups} is the live concatenation of the inners'
    C{param_groups} (the same dictionary objects), so PyTorch Lightning
    can drive several optimizer configurations through its automatic
    optimization path: one C{step}, one gradient-clipping pass over the
    union of the groups, one GradScaler slot.

    The parameter partition is static: groups are never added, removed,
    or moved after construction. Freezing and unfreezing are expressed
    through C{requires_grad} only; the inner optimizers natively skip
    parameters whose gradient is C{None}.

    Deliberately does not call C{Optimizer.__init__}: the base
    initializer would build its own parameter groups. The narrow
    contract Lightning relies on (the C{Optimizable} protocol,
    C{step(closure)}, C{zero_grad}, C{state_dict}/C{load_state_dict})
    is implemented directly instead.
    """

    STATE_DICT_FORMAT = "luxonis_composite"

    def __init__(self, inners: Sequence[Optimizer]):
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
        return self._inners

    @property  # type: ignore[override]
    def param_groups(self) -> list[dict[str, Any]]:
        # Recomputed on access on purpose: `Optimizer.load_state_dict`
        # replaces the inner group dictionaries, so a stored
        # concatenation would go stale after a checkpoint restore.
        return [
            group for inner in self._inners for group in inner.param_groups
        ]

    @param_groups.setter
    def param_groups(self, value: Any) -> None:
        raise TypeError(
            "`CompositeOptimizer.param_groups` cannot be replaced; "
            "the parameter partition is fixed at construction."
        )

    @property  # type: ignore[override]
    def state(self) -> _CompositeState:
        return self._state_view

    @state.setter
    def state(self, value: Any) -> None:
        raise TypeError(
            "`CompositeOptimizer.state` cannot be replaced; it is a "
            "view over the inner optimizers' states."
        )

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Callable[[], Any] | None = None
    ) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for inner in self._inners:
            inner.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for inner in self._inners:
            inner.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        raise RuntimeError(
            "`CompositeOptimizer` is a fixed partition of the model "
            "parameters; groups cannot be added after construction."
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.STATE_DICT_FORMAT,
            "version": 1,
            "optimizers": [type(inner).__name__ for inner in self._inners],
            "inners": [inner.state_dict() for inner in self._inners],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
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
    """Inner optimizers of a (possibly composite) optimizer sequence.

    Identity for plain optimizers, so callers can treat the single-
    optimizer bypass and the composite path uniformly.
    """
    unwrapped: list[Optimizer] = []
    for optimizer in optimizers:
        if isinstance(optimizer, CompositeOptimizer):
            unwrapped.extend(optimizer.inner_optimizers)
        else:
            unwrapped.append(optimizer)
    return unwrapped
