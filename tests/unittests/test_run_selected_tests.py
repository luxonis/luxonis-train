import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
runner = importlib.import_module("run_selected_tests")
pytest_plugins = ["pytester"]


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    # CI puts pytester projects below tests/tmp. Give them their own root
    # and configuration instead of inheriting our test IDs and basetemp.
    pytester.makeini("[pytest]\n")
    pytester.makeconftest(
        '"""Markers for the synthetic suite."""\n'
        "def pytest_configure(config):\n"
        '    config.addinivalue_line("markers", "misc: GPU suite")\n'
        '    config.addinivalue_line("markers", "cpu: CPU suite")\n'
    )
    pytester.makepyfile(
        test_models=(
            '"""Synthetic tests."""\n'
            "import pytest\n"
            "\n"
            "@pytest.mark.misc\n"
            '@pytest.mark.parametrize("model", ["light", "heavy"])\n'
            "def test_model(model):\n"
            "    pass\n"
            "\n"
            "@pytest.mark.misc\n"
            "class TestModels:\n"
            "    class TestNested:\n"
            "        def test_forward(self):\n"
            "            pass\n"
            "\n"
            "@pytest.mark.cpu\n"
            "def test_cpu():\n"
            '    raise AssertionError("CPU tests must not run in the GPU suite")\n'
        )
    )
    pytester.makepyfile(
        test_other=(
            '"""Synthetic tests."""\n'
            "import pytest\n"
            "\n"
            "@pytest.mark.misc\n"
            "def test_other():\n"
            "    pass\n"
        )
    )
    return pytester


@pytest.mark.parametrize(
    ("selection", "passed", "deselected"),
    [
        ({"test_models.py::test_model[light]"}, 1, 4),
        ({"test_models.py::test_model"}, 2, 3),
        ({"test_models.py::TestModels::TestNested::test_forward"}, 1, 4),
        ({"test_models.py"}, 3, 2),
        ({"test_models.py", "test_models.py::test_model[light]"}, 3, 2),
        (set(), 4, 1),
    ],
)
def test_current_selection(
    suite: pytest.Pytester,
    selection: set[str],
    passed: int,
    deselected: int,
):
    result = suite.inline_run(
        "-m", "misc", plugins=[runner.TestSelection(selection)]
    )
    result.assertoutcome(passed=passed)
    assert (
        sum(
            len(call.items or [])
            for call in result.getcalls("pytest_deselected")
        )
        == deselected
    )


@pytest.mark.parametrize(
    "stale",
    [
        "test_deleted.py::test_old",
        "test_models.py::test_renamed",
        "test_models.py::test_model[old-parameter]",
        "test_models.py::TestModels::TestOld::test_forward",
        "test_models.py::test_cpu",
    ],
)
def test_stale_map_runs_the_whole_current_suite(
    suite: pytest.Pytester, stale: str
):
    result = suite.inline_run(
        "-m",
        "misc",
        plugins=[runner.TestSelection({stale, "test_other.py::test_other"})],
    )
    result.assertoutcome(passed=4)


def test_rename_and_source_change_with_a_real_coverage_map(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    selector = importlib.import_module("select_tests")
    package = suite.path / "luxonis_train"
    package.mkdir()
    (package / "__init__.py").write_text("")
    source = package / "head.py"
    source.write_text("def forward():\n    return 1\n")
    old_test = suite.makepyfile(
        test_original=(
            '"""Synthetic tests."""\n'
            "from luxonis_train.head import forward\n"
            "def test_forward():\n"
            "    assert forward() > 0\n"
        )
    )
    # Use a subprocess so the synthetic package cannot replace the project's
    # luxonis_train imports in the parent test session.
    with monkeypatch.context() as env:
        env.setenv("COVERAGE_FILE", str(suite.path / ".coverage"))
        result = suite.runpytest_subprocess(
            str(old_test), "--cov=luxonis_train", "--cov-context=test"
        )
    result.assert_outcomes(passed=1)
    monkeypatch.chdir(suite.path)
    test_map = selector.build("misc", ".coverage")
    assert "test_original.py::test_forward" in test_map["tests"]
    maps = {
        "misc": {
            test: set(scopes) for test, scopes in test_map["tests"].items()
        }
    }
    old_test.rename(suite.path / "test_renamed.py")
    before = source.read_text()
    after = before.replace("return 1", "return 2")
    source.write_text(after)
    picks = selector.tests_for_changes(
        "luxonis_train/head.py", selector.changed_scopes(before, after), maps
    )
    assert picks is not None
    selection = suite.path / "selected.txt"
    selection.write_text("\n".join(selector.suite_arguments(picks, "misc")))
    assert runner.__file__ is not None
    result = suite.run(
        sys.executable,
        str(Path(runner.__file__).resolve()),
        str(selection),
        "-q",
        "test_renamed.py",
    )
    result.stdout.fnmatch_lines(
        ["*Stale test map; running the whole suite*", "*1 passed*"]
    )
    assert result.ret == 0


def test_selection_does_not_hide_a_failure(suite: pytest.Pytester):
    suite.makepyfile(
        test_broken=(
            '"""Synthetic tests."""\n'
            "import pytest\n"
            "@pytest.mark.misc\n"
            "def test_broken():\n"
            "    assert False\n"
        )
    )
    result = suite.inline_run(
        "-m",
        "misc",
        plugins=[runner.TestSelection({"test_broken.py::test_broken"})],
    )
    result.assertoutcome(failed=1)


def test_main_passes_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    selection = tmp_path / "selected.txt"
    selection.write_text("test_a.py::test_one\ntest_b.py\n")

    def main(args: list[str], plugins: list) -> int:
        assert args == ["-m", "misc", "--cov"]
        assert plugins[0].selections == {"test_a.py::test_one", "test_b.py"}
        return 1

    monkeypatch.setattr(runner.pytest, "main", main)
    assert runner.main(str(selection), "-m", "misc", "--cov") == 1
