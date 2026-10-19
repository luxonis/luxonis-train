import importlib
import sys
from pathlib import Path
from textwrap import dedent

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
select_tests = importlib.import_module("select_tests")

SOURCE = dedent('''
    """Module."""
    import torch

    SCALE = 2


    class Head:
        """A head."""

        channels = 8

        @property
        def size(self):
            return 1

        @size.setter
        def size(self, value):
            self._size = value

        def forward(self, x):
            def inner(y):
                return y * SCALE

            return inner(x)
''')

MAPS = {
    "misc": {
        "tests/integration/test_a.py::test_forward": {
            "luxonis_train/head.py::Head.forward",
            "luxonis_train/head.py::Head.forward.inner",
        },
        "tests/integration/test_a.py::test_size": {
            "luxonis_train/head.py::Head.size",
        },
        "tests/integration/test_a.py::test_other": {
            "luxonis_train/other.py::run",
        },
        "": {"luxonis_train/loader.py::Loader.__getitem__"},
    }
}


def changes(old: str, new: str) -> dict[str, str]:
    return select_tests.changed_scopes(SOURCE, SOURCE.replace(old, new))


def test_docstrings_and_comments_change_no_scope():
    assert changes('"""A head."""', '"""Another head."""  # note') == {}


def test_method_body_changes_only_the_method():
    assert changes("return inner(x)", "return inner(x) + 1") == {
        "Head.forward": "def"
    }


def test_nested_function_changes_only_itself():
    assert changes("y * SCALE", "y + SCALE") == {"Head.forward.inner": "def"}


def test_property_setter_changes_the_property():
    assert changes("self._size = value", "self._size = -value") == {
        "Head.size": "def"
    }


def test_class_attribute_changes_the_class():
    assert changes("channels = 8", "channels = 16") == {"Head": "class"}


def test_module_constant_changes_the_module():
    assert changes("SCALE = 2", "SCALE = 3") == {"": "module"}


def test_scope_of_a_line():
    scope = select_tests.scope_at(SOURCE)
    lines = SOURCE.splitlines()
    assert scope(lines.index("            return y * SCALE") + 1) == (
        "Head.forward.inner",
        True,
    )
    assert scope(lines.index("    @size.setter") + 1) == ("Head.size", False)
    assert scope(lines.index("    channels = 8") + 1) == ("Head", False)
    assert scope(lines.index("SCALE = 2") + 1) == ("", False)


def test_changed_method_selects_the_tests_that_ran_it():
    picks = select_tests.tests_for_changes(
        "luxonis_train/head.py", {"Head.forward": "def"}, MAPS
    )
    assert picks == {("misc", "tests/integration/test_a.py::test_forward")}


def test_changed_class_selects_the_tests_that_ran_its_file():
    picks = select_tests.tests_for_changes(
        "luxonis_train/head.py", {"Head": "class"}, MAPS
    )
    assert picks == {
        ("misc", "tests/integration/test_a.py::test_forward"),
        ("misc", "tests/integration/test_a.py::test_size"),
    }


def test_changed_class_that_no_test_ran_selects_every_test():
    picks = select_tests.tests_for_changes(
        "luxonis_train/unused.py", {"Unused": "class"}, MAPS
    )
    assert picks is None


def test_changed_reexport_init_selects_no_test():
    picks = select_tests.tests_for_changes(
        "luxonis_train/nodes/__init__.py", {"": "module"}, MAPS
    )
    assert picks == set()


def test_added_file_that_no_test_ran_selects_every_test():
    picks = select_tests.pick_source("A", "luxonis_train/new.py", "", MAPS)
    assert picks is None


def test_added_file_selects_the_tests_that_ran_it():
    picks = select_tests.pick_source("A", "luxonis_train/head.py", "", MAPS)
    assert picks == {
        ("misc", "tests/integration/test_a.py::test_forward"),
        ("misc", "tests/integration/test_a.py::test_size"),
    }


def test_function_that_ran_outside_a_test_selects_every_test():
    picks = select_tests.tests_for_changes(
        "luxonis_train/loader.py", {"Loader.__getitem__": "def"}, MAPS
    )
    assert picks is None


def test_test_files_select_their_suites():
    assert select_tests.pick_test_path(
        "tests/integration/test_combinations.py"
    ) == {("combinations", "tests/integration/test_combinations.py")}
    assert select_tests.pick_test_path("tests/unittests/test_x.py") == set()


def test_deleted_test_file_selects_no_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    path = "tests/integration/test_combinations.py"
    assert select_tests.pick_test_path(path) == set()


def test_conftest_selects_every_test():
    assert select_tests.pick_test_path("tests/conftest.py") is None
