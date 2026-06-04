"""Validate structured documentation sections for nodes and attached modules."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
NODE_ROOT = ROOT / "luxonis_train" / "nodes"
ATTACHED_ROOT = ROOT / "luxonis_train" / "attached_modules"

NODE_BASES = {"BaseNode"}
ATTACHED_BASES = {"BaseLoss", "BaseMetric", "BaseVisualizer"}

NODE_SKIP = {"BaseNode", "BaseHead", "BaseDetectionHead"}
ATTACHED_SKIP = {"BaseLoss", "BaseMetric", "BaseVisualizer"}
REGISTERED_ATTACHED_FACTORIES = {"ConfusionMatrix", "MeanAveragePrecision"}

NODE_MARKERS = [
    "Metadata:",
    "Provenance:",
    "Variants:",
    "- Node type:",
    "- Registry name:",
    "- Task:",
    "- Attach index:",
    "- Inputs:",
    "- Outputs:",
    "- Source:",
    "- License:",
    "- Implementation notes:",
]
ATTACHED_MARKERS = [
    "Metadata:",
    "Provenance:",
    "- Module type:",
    "- Registry name:",
    "- Task:",
    "- Attached node types:",
    "- Inputs:",
    "- Outputs:",
    "- Source:",
    "- License:",
    "- Implementation notes:",
]
FORBIDDEN_SECTION_NAMES = [
    "Node metadata",
    "Model provenance",
    "Attached module metadata",
    "Module provenance",
]
FORBIDDEN_VARIANT_TEXT = [
    "csv-table:: Variant",
    "Variant parameters",
    "Variant layer parameters",
]
COMPACT_VARIANT_PARAM_RE = re.compile(r"- ``[^`:=]+=.*``")
UNQUOTED_VARIANT_RE = re.compile(
    r"^        - ``(?!None``:|\")[^`]+``:", re.MULTILINE
)
UNQUOTED_ALIAS_RE = re.compile(r"Aliases: ``(?!\")[^`]+``")


@dataclass(frozen=True)
class ClassInfo:
    path: Path
    name: str
    lineno: int
    bases: tuple[str, ...]
    docstring: str


def base_name(base: ast.expr) -> str | None:
    """Return the unqualified base class name for an AST base expression."""
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Subscript):
        return base_name(base.value)
    if isinstance(base, ast.Call):
        return base_name(base.func)
    return None


def collect_classes(root: Path) -> list[ClassInfo]:
    """Collect class definitions under ``root``."""
    classes: list[ClassInfo] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = tuple(
                name for base in node.bases if (name := base_name(base))
            )
            classes.append(
                ClassInfo(
                    path=path,
                    name=node.name,
                    lineno=node.lineno,
                    bases=bases,
                    docstring=ast.get_docstring(node) or "",
                )
            )
    return classes


def transitive_subclasses(
    classes: Iterable[ClassInfo], root_bases: set[str]
) -> set[str]:
    """Return class names that inherit from any name in ``root_bases``."""
    bases_by_name = {cls.name: cls.bases for cls in classes}
    family = set(root_bases)
    changed = True
    while changed:
        changed = False
        for name, bases in bases_by_name.items():
            if name in family or not any(base in family for base in bases):
                continue
            family.add(name)
            changed = True
    return family


def rel(path: Path) -> str:
    """Return a repository-relative path for messages."""
    return path.relative_to(ROOT).as_posix()


def missing_markers(docstring: str, markers: Iterable[str]) -> list[str]:
    """Return schema markers that are absent from ``docstring``."""
    return [marker for marker in markers if marker not in docstring]


def check_structured_docstrings() -> list[str]:
    """Validate required metadata and provenance sections."""
    errors: list[str] = []

    node_classes = collect_classes(NODE_ROOT)
    node_family = transitive_subclasses(node_classes, NODE_BASES)
    for cls in node_classes:
        if (
            cls.name not in node_family
            or cls.name in NODE_SKIP
            or cls.name.startswith("_")
        ):
            continue
        missing = missing_markers(cls.docstring, NODE_MARKERS)
        if missing:
            errors.append(
                f"{rel(cls.path)}:{cls.lineno}: {cls.name} is missing "
                f"node doc markers: {', '.join(missing)}"
            )

    attached_classes = collect_classes(ATTACHED_ROOT)
    attached_family = transitive_subclasses(attached_classes, ATTACHED_BASES)
    for cls in attached_classes:
        should_check = (
            cls.name in attached_family
            and cls.name not in ATTACHED_SKIP
            and not cls.name.startswith("_")
        ) or cls.name in REGISTERED_ATTACHED_FACTORIES
        if not should_check:
            continue
        missing = missing_markers(cls.docstring, ATTACHED_MARKERS)
        if missing:
            errors.append(
                f"{rel(cls.path)}:{cls.lineno}: {cls.name} is missing "
                f"attached module doc markers: {', '.join(missing)}"
            )

    return errors


def check_forbidden_section_names() -> list[str]:
    """Reject old long-form schema section names."""
    errors: list[str] = []
    for root in [NODE_ROOT, ATTACHED_ROOT, ROOT / "docs"]:
        for path in sorted(root.rglob("*.py" if root != ROOT / "docs" else "*.md")):
            text = path.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_SECTION_NAMES:
                if forbidden in text:
                    errors.append(
                        f"{rel(path)}: contains old section name "
                        f"{forbidden!r}; use 'Metadata' or 'Provenance'"
                    )
    return errors


def check_variant_format() -> list[str]:
    """Validate the human-readable nested node variant format."""
    errors: list[str] = []
    for path in sorted(NODE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_VARIANT_TEXT:
            if forbidden in text:
                errors.append(
                    f"{rel(path)}: contains forbidden variant marker "
                    f"{forbidden!r}"
                )
        if COMPACT_VARIANT_PARAM_RE.search(text):
            errors.append(
                f"{rel(path)}: contains compact ``key=value`` variant "
                "parameter bullets; use ``key``: ``value``"
            )
        if UNQUOTED_VARIANT_RE.search(text):
            errors.append(
                f"{rel(path)}: contains unquoted variant keys; use "
                '``"variant"`` or ``None``'
            )
        if UNQUOTED_ALIAS_RE.search(text):
            errors.append(
                f"{rel(path)}: contains unquoted variant aliases; use "
                '``"alias"`` or None'
            )
    return errors


def main() -> int:
    """Run all documentation schema checks."""
    errors = [
        *check_structured_docstrings(),
        *check_forbidden_section_names(),
        *check_variant_format(),
    ]
    if errors:
        print("Documentation schema check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
