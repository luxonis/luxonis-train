# ruff: noqa: TRY004

"""Torch-facing helpers on top of the shape contract grammar.

The grammar itself lives in `tools.shape_contract`, which the docs build
uses as well. This module only knows about `torch`, about Google-style
docstring sections, and about matching real tensors against a parsed
contract.
"""

import ast
import importlib
import inspect
import math
import pkgutil
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from torch import Tensor, nn

import luxonis_train.nodes
from tools.shape_contract import (
    ShapeContract,
    ShapeContractError,
    ShapeError,
    ShapeSpec,
    unwrap_optional,
)
from tools.shape_contract import parse_shape_contract as parse_contract_source

__all__ = [
    "ShapeContract",
    "ShapeContractError",
    "ShapeError",
    "ShapeSpec",
    "assert_matches_contract",
    "discover_module_types",
    "get_forward_owner",
    "parse_forward_arguments",
    "parse_forward_returns",
    "parse_shape_contract",
    "validate_constraints",
]


def discover_module_types() -> list[type[nn.Module]]:
    """Discover every repository-defined module in the nodes package."""
    package = luxonis_train.nodes
    for module_info in pkgutil.walk_packages(
        package.__path__, prefix=f"{package.__name__}."
    ):
        importlib.import_module(module_info.name)

    # `__subclasses__` only sees classes that are alive, so a module
    # defined inside a function body is invisible until that function
    # runs. Nodes are expected to be defined at module level.
    discovered: set[type[nn.Module]] = set()
    pending = [nn.Module]
    visited: set[type[nn.Module]] = set()
    while pending:
        parent = pending.pop()
        if parent in visited:
            continue
        visited.add(parent)
        for child in parent.__subclasses__():
            pending.append(child)
            if child.__module__.startswith("luxonis_train.nodes"):
                discovered.add(child)
    return sorted(
        discovered,
        key=lambda item: (item.__module__, item.__qualname__),
    )


def get_forward_owner(module_type: type[nn.Module]) -> type[nn.Module] | None:
    """Return the repository class that defines a module's forward method."""
    owner = next(
        parent
        for parent in module_type.__mro__
        if "forward" in parent.__dict__
    )
    if not owner.__module__.startswith("luxonis_train.nodes"):
        return None
    return owner


def parse_shape_contract(module_type: type[nn.Module]) -> ShapeContract:
    """Parse the structured shape contract from a module's forward method."""
    owner, lines = _forward_doc(module_type)
    contract = parse_contract_source(
        lines, context=f"{owner.__qualname__}.forward"
    )
    return replace(contract, errors=_parse_forward_errors(lines))


def parse_forward_arguments(module_type: type[nn.Module]) -> dict[str, str]:
    """Parse the Google-style Args section of a forward method."""
    _, lines = _forward_doc(module_type)
    return _parse_google_entries(lines, "Args")


def parse_forward_returns(module_type: type[nn.Module]) -> str:
    """Parse the natural-language Google-style Returns section."""
    _, lines = _forward_doc(module_type)
    try:
        start = lines.index("Returns:") + 1
    except ValueError:
        return ""
    description: list[str] = []
    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        if line.strip():
            description.append(line.strip())
    return " ".join(description)


def _forward_doc(
    module_type: type[nn.Module],
) -> tuple[type[nn.Module], list[str]]:
    owner = get_forward_owner(module_type)
    if owner is None:
        raise AssertionError(
            f"{module_type.__qualname__} has no repository-defined "
            "forward method"
        )
    docstring = getattr(owner.__dict__["forward"], "__doc__", None)
    if not docstring:
        raise AssertionError(f"{owner.__qualname__}.forward has no docstring")
    return owner, inspect.cleandoc(docstring).splitlines()


def _parse_google_entries(lines: list[str], section: str) -> dict[str, str]:
    try:
        start = lines.index(f"{section}:") + 1
    except ValueError:
        return {}
    entries: dict[str, str] = {}
    current: str | None = None
    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        match = re.match(
            r" {4}(?P<name>\*{0,2}\w+)(?: \([^)]*\))?:\s*(?P<text>.*)",
            line,
        )
        if match:
            name = match.group("name")
            assert isinstance(name, str)
            current_name = name.lstrip("*")
            entries[current_name] = match.group("text").strip()
            current = current_name
        elif current is not None and line.strip():
            entries[current] = f"{entries[current]} {line.strip()}".strip()
    return entries


def _parse_forward_errors(lines: list[str]) -> tuple[ShapeError, ...]:
    entries = _parse_google_entries(lines, "Raises")
    errors: list[ShapeError] = []
    for raises, description in entries.items():
        match = re.fullmatch(
            r'If (?P<when>.+?)\. The message contains "(?P<match>.+)"\.',
            description,
        )
        if match:
            errors.append(
                ShapeError(
                    when=match.group("when"),
                    raises=raises,
                    match=match.group("match"),
                )
            )
    return tuple(errors)


def assert_matches_contract(
    value: Any,
    specification: ShapeSpec,
    bindings: dict[str, int | Sequence[int]],
    *,
    path: str,
) -> None:
    """Assert that a nested tensor value matches a shape specification."""
    _, specification = unwrap_optional(specification)

    if isinstance(value, Tensor):
        if isinstance(specification, dict) and set(specification) == {"shape"}:
            shape_symbol = specification["shape"]
            if (
                not isinstance(shape_symbol, str)
                or not shape_symbol.isidentifier()
            ):
                raise AssertionError(f"{path} has an invalid shape symbol")
            actual_shape = tuple(int(dimension) for dimension in value.shape)
            expected_shape = bindings.get(shape_symbol)
            if expected_shape is None:
                bindings[shape_symbol] = actual_shape
            elif not isinstance(expected_shape, Sequence):
                raise AssertionError(f"{shape_symbol} is not a tensor shape")
            elif tuple(expected_shape) != actual_shape:
                raise AssertionError(
                    f"{path} has shape {actual_shape}, "
                    f"expected {expected_shape}"
                )
            return
        if not isinstance(specification, list):
            raise AssertionError(
                f"{path} must have a tensor shape specification"
            )
        if len(value.shape) != len(specification):
            raise AssertionError(
                f"{path} has rank {value.ndim}, expected {len(specification)}"
            )
        for index, (actual, expected) in enumerate(
            zip(value.shape, specification, strict=True)
        ):
            _assert_dimension(
                int(actual), expected, bindings, f"{path}.shape[{index}]"
            )
        return

    if isinstance(value, Mapping):
        if not isinstance(specification, dict):
            raise AssertionError(f"{path} must have an object specification")
        if "repeat" in specification:
            raise AssertionError(
                f"{path} is a mapping, not a repeated sequence"
            )
        tensor_keys = {
            str(key) for key, item in value.items() if _contains_tensor(item)
        }
        required = {
            key
            for key, item in specification.items()
            if not unwrap_optional(item)[0]
        }
        if not required <= tensor_keys <= set(specification):
            raise AssertionError(
                f"{path} tensor fields are {sorted(tensor_keys)}, "
                f"documented fields are {sorted(specification)} "
                f"of which {sorted(required)} are required"
            )
        for key, child_specification in specification.items():
            if key not in tensor_keys:
                continue
            assert_matches_contract(
                value[key],
                child_specification,
                bindings,
                path=f"{path}.{key}",
            )
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if isinstance(specification, dict) and "repeat" in specification:
            length = _evaluate_dimension(specification["repeat"], bindings)
            if len(value) != length:
                raise AssertionError(
                    f"{path} has length {len(value)}, expected {length}"
                )
            item_specification = specification.get("item")
            if item_specification is None:
                raise AssertionError(
                    f"{path} repeat specification has no item"
                )
            for index, item in enumerate(value):
                indexed_bindings = bindings | {"i": index}
                assert_matches_contract(
                    item,
                    item_specification,
                    indexed_bindings,
                    path=f"{path}[{index}]",
                )
            return
        if not isinstance(specification, list) or len(value) != len(
            specification
        ):
            raise AssertionError(
                f"{path} has an undocumented sequence structure"
            )
        for index, (item, child_specification) in enumerate(
            zip(value, specification, strict=True)
        ):
            assert_matches_contract(
                item,
                child_specification,
                bindings,
                path=f"{path}[{index}]",
            )
        return

    raise AssertionError(f"{path} is not a documented tensor structure")


def validate_constraints(
    constraints: Sequence[str], bindings: dict[str, int | Sequence[int]]
) -> None:
    """Assert that generated bindings satisfy the documented constraints."""
    for constraint in constraints:
        if not _evaluate_expression(constraint, bindings):
            raise AssertionError(
                f"Unsatisfied documented constraint: {constraint}"
            )


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_tensor(item) for item in value)
    return False


def _assert_dimension(
    actual: int,
    specification: ShapeSpec,
    bindings: dict[str, int | Sequence[int]],
    path: str,
) -> None:
    if specification == "*":
        return
    if isinstance(specification, int):
        expected = specification
    elif isinstance(specification, str) and specification.isidentifier():
        bound = bindings.get(specification)
        if bound is None:
            bindings[specification] = actual
            return
        if not isinstance(bound, int):
            raise AssertionError(f"{specification} is not a scalar dimension")
        expected = bound
    else:
        expected = _evaluate_dimension(specification, bindings)
    if actual != expected:
        raise AssertionError(f"{path} is {actual}, expected {expected}")


def _evaluate_dimension(
    expression: ShapeSpec, bindings: dict[str, int | Sequence[int]]
) -> int:
    if isinstance(expression, int):
        return expression
    if not isinstance(expression, str):
        raise AssertionError(f"Invalid dimension expression: {expression!r}")
    value = _evaluate_expression(expression, bindings)
    if not isinstance(value, int):
        raise AssertionError(
            f"Dimension expression is not an integer: {expression}"
        )
    return value


def _evaluate_expression(
    expression: str, bindings: dict[str, int | Sequence[int]]
) -> int | bool:
    tree = ast.parse(expression, mode="eval")
    return _evaluate_node(tree.body, bindings)


def _evaluate_node(
    node: ast.AST, bindings: dict[str, int | Sequence[int]]
) -> int | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return node.value
    if isinstance(node, ast.Name):
        try:
            value = bindings[node.id]
        except KeyError as exc:
            raise AssertionError(f"Unbound shape symbol: {node.id}") from exc
        if isinstance(value, Sequence):
            raise AssertionError(f"Shape symbol {node.id} requires an index")
        return value
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        values = bindings.get(node.value.id)
        if not isinstance(values, Sequence):
            raise AssertionError(
                f"Shape symbol {node.value.id} is not indexable"
            )
        index = _evaluate_node(node.slice, bindings)
        if not isinstance(index, int):
            raise AssertionError("Shape index must be an integer")
        return int(values[index])
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, bindings)
        right = _evaluate_node(node.right, bindings)
        if not isinstance(left, int) or not isinstance(right, int):
            raise AssertionError("Shape arithmetic only accepts integers")
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            raise AssertionError("Unsupported shape arithmetic")
        return operation()
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == len(node.comparators) == 1
    ):
        left = _evaluate_node(node.left, bindings)
        right = _evaluate_node(node.comparators[0], bindings)
        comparisons = {
            ast.Eq: lambda: left == right,
            ast.NotEq: lambda: left != right,
            ast.Lt: lambda: left < right,
            ast.LtE: lambda: left <= right,
            ast.Gt: lambda: left > right,
            ast.GtE: lambda: left >= right,
        }
        comparison = comparisons.get(type(node.ops[0]))
        if comparison is None:
            raise AssertionError("Unsupported shape comparison")
        return comparison()
    if isinstance(node, ast.BoolOp):
        values = [
            bool(_evaluate_node(value, bindings)) for value in node.values
        ]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ceil_div"
        and len(node.args) == 2
    ):
        numerator = _evaluate_node(node.args[0], bindings)
        denominator = _evaluate_node(node.args[1], bindings)
        if isinstance(numerator, int) and isinstance(denominator, int):
            return math.ceil(numerator / denominator)
    raise AssertionError(f"Unsupported shape expression: {ast.unparse(node)}")
