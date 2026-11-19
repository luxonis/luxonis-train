import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SUITES = {"predefined_light", "predefined_heavy", "combinations", "misc"}
DRAFT_SUITES = SUITES - {"predefined_heavy"}


@pytest.fixture
def plan_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    git = shutil.which("git")
    assert git is not None
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("select_tests.py", "only_docstrings_changed.py"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    package = tmp_path / "luxonis_train"
    package.mkdir()
    source = package / "head.py"
    source.write_text("def forward():\n    return 1\n")
    for args in (
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "base",
        ),
    ):
        subprocess.run([git, *args], check=True, capture_output=True)
    base = subprocess.check_output(
        [git, "rev-parse", "HEAD"], text=True
    ).strip()
    source.write_text("def forward():\n    return 2\n")
    subprocess.run([git, "add", "."], check=True)
    subprocess.run(
        [
            git,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "change",
        ],
        check=True,
    )
    monkeypatch.setenv("BASE_SHA", base)
    monkeypatch.setenv("HEAD_REF", "draft-branch")
    monkeypatch.setenv("BASE_REF", "main")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/train")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))

    for suite in SUITES:
        # The preferred branch is on the second artifact page. The first
        # page's map is newer, but comes from an unrelated branch.
        def artifact(artifact_id: int, branch: str, date: str) -> dict:
            return {
                "id": artifact_id,
                "expired": False,
                "created_at": date,
                "workflow_run": {
                    "head_branch": branch,
                    "head_repository_id": 1,
                    "repository_id": 1,
                },
            }

        pages = [
            {"artifacts": [artifact(2, "unrelated", "2026-11-19")]},
            {"artifacts": [artifact(1, "draft-branch", "2026-11-18")]},
        ]
        (tmp_path / f"{suite}-pages.json").write_text(json.dumps(pages))
    with zipfile.ZipFile(tmp_path / "artifact.zip", "w") as archive:
        for suite in SUITES:
            archive.writestr(
                f"test-map-{suite}.json",
                json.dumps(
                    {
                        "suite": suite,
                        "tests": {
                            "tests/integration/test_model.py::test_forward": [
                                "luxonis_train/head.py::forward"
                            ]
                        },
                    }
                ),
            )
    return tmp_path


@pytest.mark.parametrize(
    ("pull_request", "draft", "docs_only", "failure", "expected"),
    [
        (True, True, False, "", "selected"),
        (True, False, False, "", "full"),
        (False, False, False, "", "full"),
        (True, True, True, "", "none"),
        (True, False, True, "", "none"),
        (True, True, False, "list", "draft-full"),
        (True, True, False, "download", "draft-full"),
    ],
)
def test_workflow_plan(
    plan_repo: Path,
    pull_request: bool,
    draft: bool,
    docs_only: bool,
    failure: str,
    expected: str,
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yaml").read_text())
    shell = workflow["jobs"]["plan"]["steps"][-1]["run"]
    # Exercise the workflow's actual shell and selector. Only GitHub's API
    # is replaced; its pagination flags and chosen artifact are checked.
    stub = r"""
    gh() {
      if [[ "$*" == *"actions/artifacts?name="* ]]; then
        [[ "$API_FAILURE" != list ]] || return 1
        [[ "$*" == *"--paginate --slurp"* ]] || return 2
        local suite="${*: -1}"
        suite="${suite#*name=test-map-}"
        suite="${suite%%&*}"
        cat "$suite-pages.json"
      else
        [[ "$API_FAILURE" != download ]] || return 1
        [[ "$*" == *"/artifacts/1/zip" ]] || return 2
        cat artifact.zip
      fi
    }
    """
    env = dict(
        os.environ,
        PULL_REQUEST=str(pull_request).lower(),
        DRAFT=str(draft).lower(),
        API_FAILURE=failure,
        PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}",
    )
    if docs_only:
        # Comparing HEAD to itself is the same no-code-change result as a
        # docstring-only diff; the AST comparison has its own tests.
        env["BASE_SHA"] = "HEAD"
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, "-e", "-o", "pipefail", "-c", stub + shell],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    outputs = dict(
        line.split("=", 1)
        for line in (plan_repo / "output").read_text().splitlines()
    )
    tests = json.loads(outputs["gpu_tests"])
    assert set(json.loads(outputs["gpu_suites"])) == set(tests)
    assert outputs["code_changed"] == str(not docs_only).lower()
    if expected == "selected":
        assert set(tests) == DRAFT_SUITES
        assert all(
            args == ["tests/integration/test_model.py::test_forward"]
            for args in tests.values()
        )
    elif expected == "none":
        assert tests == {}
    else:
        suites = DRAFT_SUITES if expected == "draft-full" else SUITES
        assert tests == {suite: [] for suite in suites}
    if failure:
        assert "::warning::" in result.stdout
