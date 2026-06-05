"""Build a local LuxonisML intersphinx inventory for pydoctor."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "_intersphinx" / "luxonis_ml"


def _find_luxonis_ml_package() -> Path:
    spec = importlib.util.find_spec("luxonis_ml")
    if spec is None or spec.submodule_search_locations is None:
        msg = (
            "Cannot build LuxonisML intersphinx inventory because "
            "`luxonis_ml` is not installed."
        )
        raise RuntimeError(msg)
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def build_inventory(output_dir: Path) -> int:
    """Build ``objects.inv`` from the installed ``luxonis_ml`` package."""
    try:
        from pydoctor.driver import main as pydoctor_main
    except ImportError as exc:
        msg = (
            "Cannot build LuxonisML intersphinx inventory because "
            "`pydoctor` is not installed."
        )
        raise RuntimeError(msg) from exc

    package_dir = _find_luxonis_ml_package()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pydoctor reads ./pyproject.toml by default. Run from an empty directory so
    # the luxonis-train pydoctor config does not require the inventory while we
    # are creating it.
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        try:
            exit_code = pydoctor_main(
                [
                    "--make-intersphinx",
                    "--docformat=google",
                    "--project-name=luxonis-ml",
                    f"--html-output={output_dir}",
                    str(package_dir),
                ]
            )
        finally:
            os.chdir(cwd)

    inventory = output_dir / "objects.inv"
    if exit_code == 0 or inventory.exists():
        return 0
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where pydoctor should write objects.inv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build_inventory(args.output_dir.resolve())


if __name__ == "__main__":
    sys.exit(main())
