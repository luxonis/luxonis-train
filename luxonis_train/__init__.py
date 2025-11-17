import sys
from typing import Final

from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.4.1"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)


# Do not run imports when first importing from within the CLI
# This is to make the CLI more responsive
if (
    "_unlocked" in locals()
    or "--source" in sys.argv
    or not sys.argv[0].endswith("/luxonis_train")
):
    import pathlib
    import warnings

    try:
        import torch

        from .attached_modules import *
        from .config.predefined_models import *
        from .core import *
        from .lightning import *
        from .loaders import *
        from .nodes import *
        from .optimizers import *
        from .schedulers import *
        from .strategies import *
        from .tasks import *
        from .utils import *
        from .utils import setup_logging

        setup_logging()
        torch.serialization.add_safe_globals(
            [
                pathlib.Path,
                pathlib.PosixPath,
                pathlib.WindowsPath,
            ]
        )

    except ImportError as e:
        warnings.warn(
            "Failed to import submodules. "
            "Some functionality of `luxonis-train` may be unavailable. "
            f"Error: `{e}`",
            stacklevel=2,
        )
else:  # pragma: no cover
    _unlocked = ...
