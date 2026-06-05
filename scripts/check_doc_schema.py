"""Validate structured documentation sections for nodes and attached
modules.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE_ROOT = ROOT / "luxonis_train" / "nodes"
ATTACHED_ROOT = ROOT / "luxonis_train" / "attached_modules"
PACKAGE_ROOT = ROOT / "luxonis_train"

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
SCHEMA_LITERAL_SINGLE_BACKTICK_RE = re.compile(r"(?<!`)`([^`]+)`(?!`)")
SCHEMA_LITERAL_NAMES = {
    "None",
    "True",
    "False",
    "train",
    "eval",
    "test",
    "export",
    "task_name",
    "dataset_dir",
    "dataset_name",
    "weights",
    "view",
    "mode",
    "node",
    "inputs",
    "outputs",
    "predictions",
    "targets",
    "images",
    "labels",
}
GOOGLE_ARG_SECTION_RE = re.compile(
    r"^\s*(Args|Arguments|Parameters|Keyword Args):\s*$"
)
GOOGLE_SECTION_RE = re.compile(r"^\s*[A-Z][A-Za-z ]+:\s*$")
GOOGLE_ARG_RE = re.compile(
    r"^\s+(?P<name>\*{0,2}[A-Za-z_][A-Za-z0-9_]*)(?:\s*\([^)]*\))?:"
)


@dataclass(frozen=True)
class ClassInfo:
    path: Path
    name: str
    lineno: int
    bases: tuple[str, ...]
    docstring: str


@dataclass(frozen=True)
class FunctionInfo:
    path: Path
    name: str
    qualname: str
    lineno: int
    args: ast.arguments
    docstring: str


def base_name(base: ast.expr) -> str | None:
    """Return the unqualified base class name for an AST base
    expression.
    """
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


def collect_functions(root: Path) -> list[FunctionInfo]:
    """Collect function and method definitions under ``root``."""
    functions: list[FunctionInfo] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Visitor(ast.NodeVisitor):
            def __init__(self, source_path: Path) -> None:
                self.source_path = source_path
                self.stack: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def _visit_function(
                self, node: ast.FunctionDef | ast.AsyncFunctionDef
            ) -> None:
                qualname = ".".join([*self.stack, node.name])
                functions.append(
                    FunctionInfo(
                        path=self.source_path,
                        name=node.name,
                        qualname=qualname,
                        lineno=node.lineno,
                        args=node.args,
                        docstring=ast.get_docstring(node) or "",
                    )
                )
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> None:
                self._visit_function(node)

        Visitor(path).visit(tree)
    return functions


def transitive_subclasses(
    classes: Iterable[ClassInfo], root_bases: set[str]
) -> set[str]:
    """Return class names that inherit from any name in
    ``root_bases``.
    """
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


def google_doc_arg_names(docstring: str) -> list[str]:
    """Extract documented parameter names from Google-style arg
    sections.
    """
    names: list[str] = []
    lines = docstring.splitlines()
    in_args = False
    section_indent = 0
    arg_indent: int | None = None
    for line in lines:
        section_match = GOOGLE_ARG_SECTION_RE.match(line)
        if section_match:
            in_args = True
            section_indent = len(line) - len(line.lstrip())
            arg_indent = None
            continue

        if not in_args:
            continue

        if (
            GOOGLE_SECTION_RE.match(line)
            and len(line) - len(line.lstrip()) <= section_indent
        ):
            in_args = False
            continue

        arg_match = GOOGLE_ARG_RE.match(line)
        if not arg_match:
            continue

        line_indent = len(line) - len(line.lstrip())
        if arg_indent is None:
            arg_indent = line_indent
        if line_indent == arg_indent:
            names.append(arg_match.group("name"))
    return names


def signature_arg_names(args: ast.arguments) -> set[str]:
    """Return valid documented parameter names for a function
    signature.
    """
    names = {
        arg.arg
        for arg in [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
        ]
    }
    names.discard("self")
    names.discard("cls")
    if args.vararg is not None:
        names.add(args.vararg.arg)
        names.add(f"*{args.vararg.arg}")
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
        names.add(f"**{args.kwarg.arg}")
    return names


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
        for path in sorted(
            root.rglob("*.py" if root != ROOT / "docs" else "*.md")
        ):
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


def check_schema_single_backtick_literals() -> list[str]:
    """Reject obvious single-backtick literals in schema docstrings."""
    errors: list[str] = []
    schema_classes = [
        *collect_classes(NODE_ROOT),
        *collect_classes(ATTACHED_ROOT),
    ]
    for cls in schema_classes:
        if not cls.docstring or "Metadata:" not in cls.docstring:
            continue
        for match in SCHEMA_LITERAL_SINGLE_BACKTICK_RE.finditer(cls.docstring):
            value = match.group(1)
            if value in SCHEMA_LITERAL_NAMES or value.startswith(('"', "'")):
                errors.append(
                    f"{rel(cls.path)}:{cls.lineno}: {cls.name} uses "
                    f"single-backtick literal `{value}` in schema docs; "
                    "use double backticks"
                )
    return errors


def check_function_doc_args_match_signatures() -> list[str]:
    """Reject documented function arguments that are absent from
    signatures.
    """
    errors: list[str] = []
    for func in collect_functions(PACKAGE_ROOT):
        if not func.docstring:
            continue
        documented = google_doc_arg_names(func.docstring)
        if not documented:
            continue
        signature_names = signature_arg_names(func.args)
        unknown = [
            name
            for name in documented
            if name not in signature_names
            and name.lstrip("*") not in signature_names
        ]
        if unknown:
            errors.append(
                f"{rel(func.path)}:{func.lineno}: {func.qualname} "
                "documents arguments not present in the signature: "
                f"{', '.join(unknown)}"
            )
    return errors


def main() -> int:
    """Run all documentation schema checks."""
    errors = [
        *check_structured_docstrings(),
        *check_forbidden_section_names(),
        *check_variant_format(),
        *check_schema_single_backtick_literals(),
        *check_function_doc_args_match_signatures(),
    ]
    if errors:
        print("Documentation schema check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
