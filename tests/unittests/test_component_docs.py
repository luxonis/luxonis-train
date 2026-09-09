import ast
import importlib
import pathlib

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
