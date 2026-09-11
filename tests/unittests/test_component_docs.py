import ast
import importlib
import inspect
import operator
import pathlib
import re
from typing import TypeAlias

import pytest
import yaml

from luxonis_train.config import Config
from luxonis_train.registry import LOSSES, METRICS, NODES, VISUALIZERS

importlib.import_module("luxonis_train.nodes")
importlib.import_module("luxonis_train.attached_modules")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = sorted(
    list((ROOT / "luxonis_train" / "nodes").rglob("*.py"))
    + list((ROOT / "luxonis_train" / "attached_modules").rglob("*.py"))
)
DIRECTIVE = ".. code-block:: yaml"
VariantValue: TypeAlias = "bool | int | float | str | tuple[VariantValue, ...] | list[VariantValue] | None"


def _yaml_blocks(doc: str) -> list[str]:
    """Extract the literal block of every ``code-block:: yaml``.

    A block runs until the first non-blank line indented no deeper than
    its directive, and the last line need not end with a newline.
    """
    lines = doc.split("\n")
    blocks = []
    for i, line in enumerate(lines):
        if line.strip() != DIRECTIVE:
            continue
        indent = len(line) - len(line.lstrip())
        body = []
        for candidate in lines[i + 1 :]:
            if not candidate.strip():
                body.append("")
                continue
            if len(candidate) - len(candidate.lstrip()) <= indent:
                break
            body.append(candidate)
        blocks.append("\n".join(body))
    return blocks


def _documented() -> list[tuple[str, str, list[dict]]]:
    """Every class docstring's example, parsed from its YAML block."""
    examples = []
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            doc = ast.get_docstring(node)
            if doc is None:
                continue
            examples.extend(
                (node.name, str(path.relative_to(ROOT)), yaml.safe_load(block))
                for block in _yaml_blocks(doc)
            )
    return examples


EXAMPLES = _documented()


def test_every_component_documents_an_example():
    documented = {name for name, _, _ in EXAMPLES}
    # Other tests register their own dummy components; only the ones
    # this package ships are documented.
    registered = {
        cls.__name__
        for registry in (NODES, LOSSES, METRICS, VISUALIZERS)
        for cls in registry._module_dict.values()
        if cls.__module__.startswith("luxonis_train.")
    }
    # Base classes carry no example; they are never named in a config.
    missing = registered - documented - {"BaseHead", "BaseDetectionHead"}
    assert not missing


@pytest.mark.parametrize(
    ("cls_name", "entries"),
    [(name, block) for name, _, block in EXAMPLES],
    ids=[f"{name}-{path}" for name, path, _ in EXAMPLES],
)
def test_example_is_a_valid_node_graph(cls_name: str, entries: list[dict]):
    """The examples must load as the ``model.nodes`` they claim to
    be.
    """
    config = Config.get_config(
        {
            "model": {"name": "docs", "nodes": _complete(entries)},
            "loader": {"params": {"dataset_name": "docs"}},
        }
    )
    mentioned = {node.name for node in config.model.nodes}
    for node in config.model.nodes:
        for attached in (*node.losses, *node.metrics, *node.visualizers):
            mentioned.add(attached.name)
    assert mentioned & _registered_names(cls_name)


def _registered_names(cls_name: str) -> set[str]:
    """Collect the config names a class is registered under."""
    return {
        name
        for registry in (NODES, LOSSES, METRICS, VISUALIZERS)
        for name, cls in registry._module_dict.items()
        if cls.__name__ == cls_name
    }


def _complete(entries: list[dict]) -> list[dict]:
    """Prepend the nodes an example refers to but does not show."""
    shown = {entry["name"] for entry in entries}
    prefix: list[dict] = []
    pending = [
        source
        for entry in entries
        for source in entry.get("inputs", [])
        if source not in shown
    ]
    while pending:
        name = pending.pop(0)
        if name in shown:
            continue
        shown.add(name)
        node: dict = {"name": name}
        variant = _default_variant(name)
        if variant is not None:
            node["variant"] = variant
        prefix.insert(0, node)
    return prefix + entries


def _default_variant(name: str) -> str | None:
    try:
        default, _ = NODES.get(name).get_variants()
    except NotImplementedError:
        return None
    return default


VARIANT_NODES = sorted(
    {
        cls.__name__: cls
        for cls in NODES._module_dict.values()
        if cls.__module__.startswith("luxonis_train.")
        and cls.__dict__.get("__doc__")
    }.items()
)


@pytest.mark.parametrize(
    ("name", "cls"), VARIANT_NODES, ids=[n for n, _ in VARIANT_NODES]
)
def test_documented_variants_match_the_code(name: str, cls: type):
    """The ``Variants`` section must agree with ``get_variants``.

    The section may leave a parameter out -- the layer tables of
    `MicroNet` and `GhostFaceNet` would bury the docstring -- but every
    parameter it does list has to be one the variant sets, with the
    value the variant sets.

    """
    documented = _documented_variants(
        inspect.cleandoc(cls.__dict__["__doc__"])
    )
    if documented is None:
        pytest.skip(f"{name} documents no variants")
    try:
        default, variants = cls.get_variants()
    except NotImplementedError:
        assert not documented, f"{name} documents variants but declares none"
        return

    assert documented, f"{name} declares variants but documents none"
    named = set(documented) | {
        alias for entry in documented.values() for alias in entry["aliases"]
    }
    assert named == set(variants)
    assert [key for key, entry in documented.items() if entry["default"]] == [
        default
    ]
    for key, entry in documented.items():
        for alias in entry["aliases"]:
            assert variants[alias] == variants[key]
        for parameter, value in entry["params"].items():
            assert parameter in variants[key], (
                f"{name}[{key}] documents `{parameter}`, which the variant "
                "does not set"
            )
            assert variants[key][parameter] == value


def _documented_variants(doc: str) -> dict[str, dict] | None:
    """Parse the ``Variants`` section, or ``None`` when there is
    none.
    """
    body = _variants_section(doc)
    if body is None:
        return None
    if any("None. Configure the node" in line for line in body):
        return {}
    return _variant_entries(body)


def _variants_section(doc: str) -> list[str] | None:
    lines = doc.split("\n")
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "Variants:"),
        None,
    )
    if start is None:
        return None
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith("    "):
            break
        body.append(line)
    return body


def _variant_entries(body: list[str]) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    current: dict | None = None
    for line in body:
        match = re.match(r"^    - ``(.+?)``:$", line)
        if match:
            current = {"default": False, "aliases": [], "params": {}}
            entries[_text(_literal(match.group(1)))] = current
            continue
        if current is not None:
            _read_variant_field(current, line)
    return entries


def _read_variant_field(entry: dict, line: str) -> None:
    match = re.match(r"^        - (Default|Aliases): ?(.*)$", line)
    if match:
        field, value = match.groups()
        if field == "Default":
            entry["default"] = value.strip() == "yes"
        elif value.strip() != "None":
            entry["aliases"] = [
                _text(_literal(alias.strip().strip("`")))
                for alias in value.split(",")
            ]
        return
    match = re.match(r"^            - ``(.+?)``: ``(.+?)``$", line)
    if match:
        entry["params"][match.group(1)] = _literal(match.group(2))


def _literal(text: str) -> VariantValue:
    """Evaluate a documented value, which may be written as ``1 /
    2``.
    """
    return _walk(ast.parse(text, mode="eval").body)


def _walk(node: ast.AST) -> VariantValue:
    if isinstance(node, ast.Constant):
        value = node.value
        assert value is None or isinstance(value, bool | int | float | str)
        return value
    if isinstance(node, ast.Tuple):
        return tuple(_walk(element) for element in node.elts)
    if isinstance(node, ast.List):
        return [_walk(element) for element in node.elts]
    return _arithmetic(node)


def _arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_arithmetic(node.operand)
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return operators[type(node.op)](
            _arithmetic(node.left), _arithmetic(node.right)
        )
    return _number(_walk(node))


def _number(value: VariantValue) -> int | float:
    assert isinstance(value, int | float)
    return value


def _text(value: VariantValue) -> str:
    assert isinstance(value, str)
    return value
