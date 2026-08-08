"""The description of a single executable shape contract case."""

from dataclasses import dataclass

from torch import nn


@dataclass
class ContractCase:
    """A module, its inputs and the symbol bindings of its contract."""

    module: nn.Module
    inputs: dict
    bindings: dict
    mode: str | None = None
    outputs: object | None = None
