"""Tell whether the changes since a base commit touch only
documentation.

The CI runs the GPU test suites only when the code changes. The script
exits with 0 when each changed file is one of these:

  - a Python file whose syntax tree without docstrings stays the same.
    Comments and formatting never reach the syntax tree;
  - C{pyproject.toml} with changes in the tables of the linters and the
    API docs only, see C{DOC_TABLES};
  - a Markdown file, C{.gitignore} or C{.pre-commit-config.yaml}.

Any other change, an added or a removed Python file included, exits
with 1 and names the file.

Usage: C{python scripts/only_docstrings_changed.py <base-commit>}

"""

import ast
import subprocess
import sys
from collections.abc import Callable

DOC_TABLES = (
    "tool.docformatter",
    "tool.pydoctor",
    "tool.pyright",
    "tool.ruff",
)
IGNORED = {".gitignore", ".pre-commit-config.yaml"}


def git(*args: str) -> str:
    """Run git and return its output."""
    return subprocess.run(
        ["git", *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def is_bare_string(statement: ast.stmt) -> bool:
    """Tell whether a statement is a string literal and nothing else."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def code_of(source: str) -> str:
    """Dump the syntax tree of C{source} without its docstrings.

    A string literal that stands as a statement of its own has no
    effect, wherever it is. This covers the docstrings of attributes
    too.

    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if isinstance(statements, list):
                setattr(
                    node,
                    field,
                    [s for s in statements if not is_bare_string(s)],
                )
    return ast.dump(tree)


def without_doc_tables(pyproject: str) -> list[str]:
    """Drop the tables of C{DOC_TABLES} from C{pyproject.toml}.

    A table header always stands on a line of its own, at the start.

    """
    kept = []
    keep = True
    for line in pyproject.splitlines():
        if line.startswith("["):
            keep = not line.strip("[] ").startswith(DOC_TABLES)
        if keep:
            kept.append(line)
    return kept


def main(base: str) -> int:
    merge_base = git("merge-base", base, "HEAD").strip()
    changes = git("diff", "--name-status", "--no-renames", merge_base, "HEAD")
    for line in changes.splitlines():
        status, path = line.split("\t")
        if path.endswith(".md") or path in IGNORED:
            continue
        compare: Callable[[str], object]
        if path == "pyproject.toml":
            compare = without_doc_tables
        elif path.endswith(".py"):
            compare = code_of
        else:
            print(f"{path}: not a documentation file")
            return 1
        if status != "M":
            print(f"{path}: added or removed")
            return 1
        before = git("show", f"{merge_base}:{path}")
        after = git("show", f"HEAD:{path}")
        if compare(before) != compare(after):
            print(f"{path}: the code changed")
            return 1
    print("Only the documentation changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
