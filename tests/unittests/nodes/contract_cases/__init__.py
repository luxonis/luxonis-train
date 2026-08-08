"""Executable cases for the documented node shape contracts.

Every group module in this package exposes two mappings. `CASES` maps a
module type to a factory building one or more `ContractCase` objects for
it, and `UNSUPPORTED` maps a module type to the reason its documented
contract cannot be executed at runtime. This module merges the groups
into the single `CONTRACT_CASES` and `UNSUPPORTED` mappings the test
suite consumes, and refuses to import when two groups claim the same
class.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

from torch import nn

from . import backbones, base, blocks, heads, necks
from .case import ContractCase

CaseFactory = Callable[[], ContractCase | list[ContractCase]]

GROUPS: tuple[ModuleType, ...] = (base, backbones, blocks, heads, necks)

__all__ = [
    "CONTRACT_CASES",
    "GROUPS",
    "UNSUPPORTED",
    "CaseFactory",
    "ContractCase",
]


def _merge(attribute: str) -> dict[type[nn.Module], Any]:
    """Merge one mapping across the groups, rejecting double claims."""
    merged: dict[type[nn.Module], Any] = {}
    owners: dict[type[nn.Module], str] = {}
    for group in GROUPS:
        name = group.__name__.rsplit(".", 1)[-1]
        entries: dict[type[nn.Module], Any] = getattr(group, attribute)
        for module_type, value in entries.items():
            owner = owners.get(module_type)
            if owner is not None:
                raise RuntimeError(
                    f"{module_type.__qualname__} is listed in {attribute} "
                    f"by both the '{owner}' and the '{name}' contract case "
                    "group. Every class must be claimed by exactly one "
                    "group, otherwise one of the two definitions is "
                    "silently dropped."
                )
            merged[module_type] = value
            owners[module_type] = name
    return merged


CONTRACT_CASES: dict[type[nn.Module], CaseFactory] = _merge("CASES")
UNSUPPORTED: dict[type[nn.Module], str] = _merge("UNSUPPORTED")
