#!/usr/bin/env python3
"""Build pydoctor API documentation for luxonis-train."""

import argparse
import ast
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

PACKAGE = "luxonis_train"
PROJECT_NAME = "luxonis-train"
PROJECT_URL = "https://github.com/luxonis/luxonis-train"
PAGES_URL = "https://luxonis.github.io/luxonis-train"
SYSTEM_CLASS = "tools.pydoctor_extensions.LuxonisTrainSystem"

# The package is mid-migration from Epytext to Google docstrings. The
# modules that have moved declare `__docformat__ = "google"` themselves,
# so the global default stays Epytext.
DOCFORMAT = "epytext"


@dataclass(frozen=True)
class VersionEntry:
    label: str
    path: str
    version: str
    kind: str


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()

    clean_output(output, repo_root)
    version = read_project_version(repo_root)

    if args.mode == "current":
        build_pydoctor(
            repo_root=repo_root,
            destination=output,
            version=version,
            base_url=None,
        )
        return

    build_pydoctor(
        repo_root=repo_root,
        destination=output / "latest",
        version=version,
        base_url=f"{PAGES_URL}/latest/",
    )
    write_site_index(output, version)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pydoctor docs for luxonis-train."
    )
    parser.add_argument(
        "--mode",
        choices=["current", "site"],
        default="current",
        help=(
            "Build only the current checkout, or build the full Pages "
            "site with a versions index."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apidocs"),
        help="Directory where the generated static site will be written.",
    )
    return parser.parse_args()


def clean_output(output: Path, repo_root: Path) -> None:
    if output == repo_root:
        raise SystemExit("Refusing to use the repository root as docs output.")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)


def build_pydoctor(
    *,
    repo_root: Path,
    destination: Path,
    version: str,
    base_url: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)

    command = [
        executable("pydoctor"),
        f"--docformat={DOCFORMAT}",
        "--project-name",
        PROJECT_NAME,
        "--project-url",
        PROJECT_URL,
        "--project-version",
        version,
        "--project-base-dir",
        str(repo_root),
        "--html-output",
        str(destination),
        "--theme",
        "readthedocs",
        f"--system-class={SYSTEM_CLASS}",
    ]
    if base_url is not None:
        command += ["--html-base-url", base_url]
    command.append(str(repo_root / PACKAGE))

    completed = subprocess.run(command, check=False, env=build_env(repo_root))
    if completed.returncode == 0:
        return
    if completed.returncode == 2 and (destination / "index.html").is_file():
        print(  # noqa: T201
            "pydoctor exited with non-fatal documentation warnings; "
            "continuing because HTML was generated."
        )
        return
    raise subprocess.CalledProcessError(completed.returncode, command)


def build_env(repo_root: Path) -> dict[str, str]:
    """Return an environment where `tools` is importable.

    Args:
        repo_root: Directory holding the `tools` package.

    Returns:
        A copy of the current environment with the repository root
        prepended to `PYTHONPATH`, so that pydoctor can import the
        class named by `--system-class`.

    """
    env = dict(os.environ)
    entries = [str(repo_root)]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def read_project_version(repo_root: Path) -> str:
    """Read `luxonis_train.__version__` without importing the package.

    Args:
        repo_root: Root of the checkout to read the version from.

    Returns:
        The declared version, or an empty string if it is not found.

    """
    init_path = repo_root / PACKAGE / "__init__.py"
    module = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in module.body:
        if (
            isinstance(node, ast.AnnAssign)
            and is_name(node.target, "__version__")
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
        if (
            isinstance(node, ast.Assign)
            and any(is_name(target, "__version__") for target in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    return ""


def is_name(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Name) and node.id == value


def write_site_index(output: Path, version: str) -> None:
    latest = VersionEntry(
        label="latest",
        path="latest/",
        version=version,
        kind="latest",
    )
    (output / "versions.json").write_text(
        json.dumps({"latest": asdict(latest), "releases": []}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    (output / "index.html").write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta http-equiv="refresh" content="0; url=latest/">\n'
        f"  <title>{PROJECT_NAME} documentation</title>\n"
        "</head>\n"
        "<body>\n"
        f'  <p><a href="latest/">{PROJECT_NAME} documentation</a></p>\n'
        "</body>\n"
        "</html>\n",
        encoding="utf-8",
    )


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"Required executable not found on PATH: {name}")
    return path


if __name__ == "__main__":
    main()
