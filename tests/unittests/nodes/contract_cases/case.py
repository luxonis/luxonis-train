"""A single executable shape contract case, and the ways to build one."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from tests.unittests.nodes.shape_contracts import Bindings


@dataclass
class ContractCase:
    """A module, its inputs and the symbol bindings of its contract."""

    module: nn.Module
    inputs: dict[str, Any]
    bindings: Bindings
    mode: str | None = None
    outputs: object | None = None


def case(
    module: nn.Module,
    inputs: dict[str, Any],
    bindings: Bindings,
    *,
    key: str = "output",
    mode: str | None = None,
) -> ContractCase:
    """Run a module now and pair its outputs with the given bindings.

    Running here rather than leaving it to the test runner is what lets
    a module be warmed up, or its inputs be kept away from a `forward`
    that consumes them. Blocks that need neither can use `feature_map`.
    """
    module.eval()
    with torch.no_grad():
        result = module(**inputs)
    return ContractCase(
        module=module,
        inputs=inputs,
        bindings=bindings,
        mode=mode,
        outputs=result if isinstance(result, Mapping) else {key: result},
    )


def feature_map(
    module: nn.Module,
    x: Tensor,
    *,
    argument: str = "x",
    out_channels: int | None = None,
    out_size: tuple[int, int] | None = None,
) -> ContractCase:
    """Build a case for a block mapping one feature map onto another.

    The bindings cover every symbol the feature-map contracts use, so
    the same call works for blocks that keep the channel count or the
    spatial size and simply do not mention the corresponding symbol.

    The runner calls the module and names the result after the contract,
    so unlike `case` this does not run anything itself.
    """
    batch, channels, height, width = x.shape
    out_height, out_width = out_size or (height, width)
    return ContractCase(
        module=module.eval(),
        inputs={argument: x},
        bindings={
            "B": batch,
            "C_in": channels,
            "C_out": channels if out_channels is None else out_channels,
            "H": height,
            "W": width,
            "H_out": out_height,
            "W_out": out_width,
        },
    )


def pyramid(
    module: nn.Module,
    inputs: dict[str, Tensor],
    shapes: list[tuple[int, int, int]],
) -> ContractCase:
    """Build a case for a node returning a feature pyramid."""
    batch, channels, height, width = next(iter(inputs.values())).shape
    return case(
        module,
        inputs,
        {
            "B": batch,
            "C_in": channels,
            "H_in": height,
            "W_in": width,
            "N": len(shapes),
            "C": tuple(shape[0] for shape in shapes),
            "H": tuple(shape[1] for shape in shapes),
            "W": tuple(shape[2] for shape in shapes),
        },
        key="features",
    )
