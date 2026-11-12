import importlib
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import coverage
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
select_tests = importlib.import_module("select_tests")
only_docstrings_changed = importlib.import_module("only_docstrings_changed")

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


def test_git_wrapper(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="output\n")

    monkeypatch.setattr(only_docstrings_changed.subprocess, "run", run)

    assert select_tests.git("show", "HEAD:file.py") == "output\n"
    assert calls == [
        (
            ["git", "show", "HEAD:file.py"],
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_build_reads_coverage_contexts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package = tmp_path / "luxonis_train"
    package.mkdir()
    source = package / "module.py"
    source.write_text("VALUE = 1\n\ndef run():\n    return 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n")

    class FakeCoverageData:
        def __init__(self, data_file: str):
            assert data_file == "coverage.data"
            self.loaded = False

        def read(self) -> None:
            self.loaded = True

        def measured_files(self) -> set[str]:
            assert self.loaded
            return {str(source), str(outside)}

        def contexts_by_lineno(self, measured: str) -> dict[int, set[str]]:
            if measured == str(outside):
                return {1: {"tests/test_outside.py::test_value|run"}}
            return {
                1: {"", "tests/test_module.py::test_run|run"},
                4: {"", "tests/test_module.py::test_run|run"},
            }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(coverage, "CoverageData", FakeCoverageData)

    assert select_tests.build("misc", "coverage.data") == {
        "suite": "misc",
        "tests": {
            "": ["luxonis_train/module.py::run"],
            "tests/test_module.py::test_run": [
                "luxonis_train/module.py::",
                "luxonis_train/module.py::run",
            ],
        },
    }


def test_test_file_suite_detection(tmp_path: Path):
    predefined = tmp_path / "test_predefined_models.py"
    predefined.write_text("")
    misc = tmp_path / "test_misc.py"
    misc.write_text("pytest.mark.gpu\n")
    cpu = tmp_path / "test_cpu.py"
    cpu.write_text("")

    assert select_tests.suites_of_test_file(predefined) == [
        "predefined_light",
        "predefined_heavy",
    ]
    assert select_tests.suites_of_test_file(misc) == ["misc"]
    assert select_tests.suites_of_test_file(cpu) == []


def test_test_helpers_select_files_that_use_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    test_file = tmp_path / "test_misc.py"
    test_file.write_text("pytest.mark.gpu\nhelper_name\n")
    monkeypatch.setattr(select_tests, "TEST_FILES", tmp_path)

    expected = {("misc", str(test_file))}
    assert select_tests.files_using("helper_name") == expected
    assert select_tests.files_using("missing") is None
    assert (
        select_tests.pick_test_path("tests/integration/helper_name.py")
        == expected
    )


def test_modified_source_compares_scopes(monkeypatch: pytest.MonkeyPatch):
    def git(*args: str) -> str:
        return (
            SOURCE
            if args[1].startswith("base:")
            else SOURCE.replace("return inner(x)", "return inner(x) + 1")
        )

    monkeypatch.setattr(select_tests, "git", git)

    assert select_tests.pick_source(
        "M", "luxonis_train/head.py", "base", MAPS
    ) == {("misc", "tests/integration/test_a.py::test_forward")}


def test_changed_config_selects_named_tests_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    test_file = tmp_path / "test_misc.py"
    test_file.write_text("pytest.mark.gpu\ndetection_light.yaml\n")
    monkeypatch.setattr(select_tests, "TEST_FILES", tmp_path)
    maps = {
        "predefined_light": {
            "tests/integration/test_predefined_models.py::test[detection_light-v1]": set()
        }
    }

    assert select_tests.pick_config(
        "luxonis_train/configs/detection_light.yaml", maps
    ) == {
        (
            "predefined_light",
            "tests/integration/test_predefined_models.py::test[detection_light-v1]",
        ),
        ("misc", str(test_file)),
    }
    assert (
        select_tests.pick_config("luxonis_train/configs/unknown.yaml", maps)
        is None
    )


def test_pick_routes_changed_files(monkeypatch: pytest.MonkeyPatch):
    config_pick = {("misc", "config")}
    source_pick = {("misc", "source")}
    test_pick = {("misc", "test")}
    monkeypatch.setattr(select_tests, "pick_config", lambda *_: config_pick)
    monkeypatch.setattr(select_tests, "pick_source", lambda *_: source_pick)
    monkeypatch.setattr(select_tests, "pick_test_path", lambda *_: test_pick)

    assert select_tests.pick("M", "README.md", "base", MAPS) == set()
    assert select_tests.pick("M", ".gitignore", "base", MAPS) == set()
    assert (
        select_tests.pick("M", ".github/dependabot.yaml", "base", MAPS)
        == set()
    )
    assert (
        select_tests.pick("M", ".github/workflows/ci.yaml", "base", MAPS)
        is None
    )
    assert select_tests.pick("M", "media/image.png", "base", MAPS) == set()
    assert (
        select_tests.pick(
            "M", "luxonis_train/configs/model.yaml", "base", MAPS
        )
        == config_pick
    )
    assert (
        select_tests.pick("M", "luxonis_train/model.py", "base", MAPS)
        == source_pick
    )
    assert (
        select_tests.pick("M", "tests/integration/helper.py", "base", MAPS)
        == test_pick
    )

    before = "[tool.ruff]\nline-length = 80\n[project]\nname = 'x'\n"
    after = "[tool.ruff]\nline-length = 90\n[project]\nname = 'x'\n"
    monkeypatch.setattr(
        select_tests,
        "git",
        lambda *args: before if args[1].startswith("base:") else after,
    )
    assert select_tests.pick("M", "pyproject.toml", "base", MAPS) == set()

    monkeypatch.setattr(
        select_tests,
        "git",
        lambda *args: (
            before
            if args[1].startswith("base:")
            else after.replace("name = 'x'", "name = 'y'")
        ),
    )
    assert select_tests.pick("M", "pyproject.toml", "base", MAPS) is None
    assert select_tests.pick("M", "Dockerfile", "base", MAPS) is None


def test_load_maps(tmp_path: Path):
    (tmp_path / "test-map-misc.json").write_text(
        json.dumps(
            {
                "suite": "misc",
                "tests": {"tests/test_a.py::test_a": ["package.py::run"]},
            }
        )
    )

    assert select_tests.load_maps(tmp_path) == {
        "misc": {
            "tests/test_a.py::test_a": {"package.py::run"},
        }
    }


def test_select_returns_every_suite_for_an_unmapped_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    def git(*args: str) -> str:
        if args[0] == "merge-base":
            return "merge-base\n"
        return "M\tDockerfile\n"

    monkeypatch.setattr(select_tests, "git", git)

    assert select_tests.select("base", str(tmp_path)) == {
        suite: [] for suite in select_tests.GPU_SUITES
    }
    assert "Dockerfile: every test" in capsys.readouterr().err


def test_select_runs_missing_suites_and_selected_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    maps = {"misc": {}}

    def git(*args: str) -> str:
        if args[0] == "merge-base":
            return "merge-base\n"
        return "M\tluxonis_train/model.py\nM\ttests/integration/test_misc.py\n"

    picks = {
        ("misc", "tests/integration/test_misc.py::test_one"),
    }
    monkeypatch.setattr(select_tests, "git", git)
    monkeypatch.setattr(select_tests, "load_maps", lambda _: maps)
    monkeypatch.setattr(
        select_tests,
        "pick",
        lambda _status, path, _base, _maps: (
            picks if path.startswith("tests/") else set()
        ),
    )

    assert select_tests.select("base", str(tmp_path)) == {
        "predefined_light": [],
        "predefined_heavy": [],
        "combinations": [],
        "misc": ["tests/integration/test_misc.py::test_one"],
    }


def test_suite_arguments_let_a_selected_file_cover_its_tests():
    picks = {
        ("misc", "tests/integration/test_a.py"),
        ("misc", "tests/integration/test_a.py::test_one"),
        ("misc", "tests/integration/test_b.py::test_two"),
    }

    assert select_tests.suite_arguments(picks, "misc") == [
        "tests/integration/test_a.py",
        "tests/integration/test_b.py::test_two",
    ]


def test_main_prints_build_and_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(
        select_tests, "build", lambda *_: {"suite": "misc", "tests": {}}
    )
    select_tests.main("build", "misc", "coverage.data")
    assert json.loads(capsys.readouterr().out) == {
        "suite": "misc",
        "tests": {},
    }

    monkeypatch.setattr(select_tests, "select", lambda *_: {"misc": []})
    select_tests.main("select", "base", "maps")
    assert json.loads(capsys.readouterr().out) == {"misc": []}
