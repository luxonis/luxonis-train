"""Select the GPU tests that a pull request can affect.

A run of a whole GPU suite writes a test map with ``build``: for each
test, the functions and the classes of the package that it ran, as
``<file>::<qualified name>``. The test ``""`` holds the functions that ran
outside of a test, such as in a data loader worker.

``select`` compares the merge base and the head without docstrings:

  - a changed function selects the tests that ran it;
  - a changed class or module body, or an added or a deleted file,
    selects the tests that ran code of the file, or every test when no
    test did. A ``__init__.py`` below ``luxonis_train/`` selects none;
  - a changed test file selects its tests, and a test helper or test
    data selects the test files that name it;
  - a changed config of ``luxonis_train/configs/`` selects the tests that
    name it;
  - a Markdown file selects none, and a function that ran outside of a
    test or any other file selects every test.

A suite without a map runs in full when the package changes.

Usage:

  - ``python scripts/select_tests.py build <suite> <coverage data file>``
  - ``python scripts/select_tests.py select <base commit> <map directory>``

``select`` prints the suites to run, each with its tests and test files.
An empty list runs the whole suite.

"""

import ast
import copy
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

from only_docstrings_changed import (
    IGNORED,
    git,
    parse_without_docstrings,
    without_doc_tables,
)

GPU_SUITES = ("predefined_light", "predefined_heavy", "combinations", "misc")
TEST_FILES = Path("tests", "integration")

Maps = dict[str, dict[str, set[str]]]
Picks = set[tuple[str, str]]
Scope = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def iter_scopes(
    node: ast.AST, prefix: str = ""
) -> Iterator[tuple[str, Scope]]:
    """Yield each function and class with its qualified name."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, Scope):
            name = prefix + child.name
            yield name, child
            yield from iter_scopes(child, f"{name}.")
        else:
            yield from iter_scopes(child, prefix)


def own_code(node: ast.AST) -> str:
    """Dump a node, with each nested function and class as its name."""
    root = copy.deepcopy(node)

    class Hide(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if node is not root and isinstance(node, Scope):
                return ast.Expr(ast.Name(node.name, ast.Load()))
            return super().visit(node)

    return ast.dump(Hide().visit(root))


def scope_codes(source: str) -> dict[str, tuple[str, str]]:
    """Map each scope of ``source`` to its kind and its own code.

    The kind is ``def``, ``class`` or ``module``, and the module is the
    scope ``""``. A property and its setter share a name, so their codes
    are joined.

    """
    tree = parse_without_docstrings(source)
    codes = {"": ("module", own_code(tree))}
    for name, node in iter_scopes(tree):
        kind = "class" if isinstance(node, ast.ClassDef) else "def"
        codes[name] = (kind, codes.get(name, ("", ""))[1] + own_code(node))
    return codes


def changed_scopes(before: str, after: str) -> dict[str, str]:
    """Map each scope that changed between two sources to its kind."""
    old, new = scope_codes(before), scope_codes(after)
    return {
        name: (old.get(name) or new[name])[0]
        for name in old.keys() | new.keys()
        if old.get(name) != new.get(name)
    }


def scope_at(source: str) -> Callable[[int], tuple[str, bool]]:
    """Return a function that gives the scope of a line of ``source``.

    The function also tells whether the line is in the body of a
    function. The other lines run when the module is imported.

    """
    ranges = []
    for name, node in iter_scopes(ast.parse(source)):
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        body = None if isinstance(node, ast.ClassDef) else node.body[0].lineno
        ranges.append((start, node.end_lineno or node.lineno, name, body))

    def scope(line: int) -> tuple[str, bool]:
        inner = [r for r in ranges if r[0] <= line <= r[1]]
        if not inner:
            return "", False
        # The innermost scope starts last.
        _, _, name, body = max(inner)
        return name, body is not None and line >= body

    return scope


def build(suite: str, data_file: str) -> dict:
    """Read the per-test coverage contexts and write the test map.

    The test ``""`` lists the functions that ran outside of a test: in a
    data loader worker, which has no test context, or on import.

    """
    from coverage import CoverageData

    data = CoverageData(data_file)
    data.read()
    root = Path.cwd()
    tests = defaultdict(set)
    for measured in data.measured_files():
        path = Path(measured)
        if not path.is_relative_to(root / "luxonis_train"):
            continue
        path = path.relative_to(root)
        scope = scope_at(path.read_text())
        for line, contexts in data.contexts_by_lineno(measured).items():
            name, in_body = scope(line)
            for context in contexts:
                # pytest-cov names a context `<test id>|<phase>`.
                test = context.partition("|")[0]
                if test or in_body:
                    tests[test].add(f"{path}::{name}")
    return {
        "suite": suite,
        "tests": {test: sorted(s) for test, s in sorted(tests.items())},
    }


def tests_running(maps: Maps, scope: str) -> Picks:
    """Select the tests that ran ``scope``, or a scope nested in it."""
    return {
        (suite, test)
        for suite, tests in maps.items()
        for test, scopes in tests.items()
        if scope in scopes or any(s.startswith(f"{scope}.") for s in scopes)
    }


def tests_in_file(maps: Maps, path: str) -> Picks:
    """Select the tests that ran code of the file ``path``."""
    return {
        (suite, test)
        for suite, tests in maps.items()
        for test, scopes in tests.items()
        if any(s.startswith(f"{path}::") for s in scopes)
    }


def tests_for_changes(
    path: str, changes: dict[str, str], maps: Maps
) -> Picks | None:
    """Select the tests for the changed scopes of a package file.

    Returns ``None`` to select every test, also when a function that ran
    outside of a test changed.

    """
    picks: Picks = set()
    for name, kind in changes.items():
        if kind == "def":
            picks |= tests_running(maps, f"{path}::{name}")
            continue
        users = tests_in_file(maps, path)
        if not users and not is_reexport_init(path):
            return None
        picks |= users
    if any(not test for _, test in picks):
        return None
    return picks


def is_reexport_init(path: str) -> bool:
    """Tell whether ``path`` is the ``__init__.py`` of a subpackage."""
    return (
        path.endswith("/__init__.py") and path != "luxonis_train/__init__.py"
    )


def suites_of_test_file(path: Path) -> list[str]:
    """Name the GPU suites of a test file.

    The rules follow ``pytest_collection_modifyitems`` in
    ``tests/conftest.py``. A deleted file has no suite.

    """
    if not path.exists():
        return []
    if path.name == "test_predefined_models.py":
        return ["predefined_light", "predefined_heavy"]
    if path.name == "test_combinations.py":
        return ["combinations"]
    if "pytest.mark.gpu" in path.read_text():
        return ["misc"]
    return []


def files_using(text: str) -> Picks | None:
    """Select the test files that contain ``text``, or every test."""
    picks = {
        (suite, str(file))
        for file in sorted(TEST_FILES.glob("test_*.py"))
        if text in file.read_text()
        for suite in suites_of_test_file(file)
    }
    return picks or None


def pick_source(status: str, path: str, base: str, maps: Maps) -> Picks | None:
    """Select the tests for a changed Python file of the package."""
    if status == "M":
        changes = changed_scopes(
            git("show", f"{base}:{path}"), git("show", f"HEAD:{path}")
        )
    else:
        changes = {"": "module"}
    return tests_for_changes(path, changes, maps)


def pick_test_path(path: str) -> Picks | None:
    """Select the tests for a changed file below ``tests/``."""
    file = Path(path)
    if file.parts[:2] == ("tests", "unittests"):
        return set()
    if file.parent == TEST_FILES and file.name.startswith("test_"):
        return {(suite, path) for suite in suites_of_test_file(file)}
    if file.name in {"conftest.py", "__init__.py"}:
        return None
    return files_using(file.stem if file.suffix == ".py" else file.name)


def pick_config(path: str, maps: Maps) -> Picks | None:
    """Select the tests that name a changed config."""
    stem = Path(path).stem
    picks = {
        (suite, test)
        for suite, tests in maps.items()
        for test in tests
        if f"[{stem}" in test
    }
    picks |= files_using(f"{stem}.yaml") or set()
    return picks or None


def pick(status: str, path: str, base: str, maps: Maps) -> Picks | None:
    """Select the tests for one changed file, or ``None`` for all."""
    if path.endswith(".md") or path in IGNORED or path.startswith("media/"):
        return set()
    if path.startswith(".github/") and path != ".github/workflows/ci.yaml":
        return set()
    if path.startswith("luxonis_train/configs/") and path.endswith(".yaml"):
        return pick_config(path, maps)
    if path.startswith("luxonis_train/") and path.endswith(".py"):
        return pick_source(status, path, base, maps)
    if path.startswith("tests/"):
        return pick_test_path(path)
    if path == "pyproject.toml" and status == "M":
        before = without_doc_tables(git("show", f"{base}:{path}"))
        after = without_doc_tables(git("show", f"HEAD:{path}"))
        return set() if before == after else None
    return None


def load_maps(directory: Path) -> Maps:
    """Read the test maps of the GPU suites from ``directory``."""
    maps = {}
    for file in directory.glob("test-map-*.json"):
        content = json.loads(file.read_text())
        maps[content["suite"]] = {
            test: set(scopes) for test, scopes in content["tests"].items()
        }
    return maps


def select(base: str, map_dir: str) -> dict[str, list[str]]:
    """Select the GPU tests for the changes since ``base``."""
    maps = load_maps(Path(map_dir))
    merge_base = git("merge-base", base, "HEAD").strip()
    changes = git("diff", "--name-status", "--no-renames", merge_base, "HEAD")
    picks: Picks = set()
    package_changed = False
    for line in changes.splitlines():
        status, path = line.split("\t")
        package_changed |= path.startswith("luxonis_train/")
        picked = pick(status, path, merge_base, maps)
        print(
            f"{path}: {'every test' if picked is None else len(picked)}",
            file=sys.stderr,
        )
        if picked is None:
            return {suite: [] for suite in GPU_SUITES}
        picks |= picked

    selection = {}
    for suite in GPU_SUITES:
        if suite not in maps and package_changed:
            selection[suite] = []
        elif args := suite_arguments(picks, suite):
            selection[suite] = args
    return selection


def suite_arguments(picks: Picks, suite: str) -> list[str]:
    """List the tests and the test files of ``suite`` to run.

    A test of a selected test file is left out, because the file runs
    it.

    """
    args = {arg for s, arg in picks if s == suite}
    files = {arg for arg in args if "::" not in arg}
    return sorted(
        a for a in args if a in files or a.split("::")[0] not in files
    )


def main(command: str, *args: str) -> None:
    if command == "build":
        print(json.dumps(build(*args), indent=1))
    else:
        print(json.dumps(select(*args)))


if __name__ == "__main__":
    main(*sys.argv[1:])
