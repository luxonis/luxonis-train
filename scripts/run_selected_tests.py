"""Run a selection against pytest's current collection.

Usage: ``python scripts/run_selected_tests.py <selection file> [pytest
args]``. The file contains one test ID or test-file path per line. If a
selection no longer exists, run the whole collected suite instead. This
covers maps from before a test rename, deletion or parametrization
change without passing stale IDs to pytest or silently dropping affected
tests.

"""

import sys
from pathlib import Path

import pytest


class TestSelection:
    """Filter the current suite only when every selection still
    exists.
    """

    def __init__(self, selections: set[str]):
        self.selections = selections

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        """Apply the selection after the suite markers have been
        applied.
        """
        if not self.selections:
            return
        matched = set()
        selected, deselected = [], []
        for item in items:
            matches = {
                arg
                for arg in self.selections
                if item.nodeid == arg
                or item.nodeid.partition("[")[0] == arg
                or item.nodeid.startswith(f"{arg}::")
            }
            matched |= matches
            (selected if matches else deselected).append(item)
        if missing := self.selections - matched:
            reporter = config.pluginmanager.get_plugin("terminalreporter")
            if reporter:
                reporter.write_line(
                    "Stale test map; running the whole suite. Missing: "
                    + ", ".join(sorted(missing))
                )
            return
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)


def main(selection_file: str, *args: str) -> int:
    """Run pytest without using cached test IDs as collection
    arguments.
    """
    selections = set(Path(selection_file).read_text().splitlines())
    return pytest.main(list(args), plugins=[TestSelection(selections)])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(*sys.argv[1:]))
