"""Train computer vision models for Luxonis cameras from a YAML config.

One config file describes the whole pipeline: the loader, the model
graph, the losses, the metrics, the visualizers, and the export
settings. The ``luxonis_train`` command and `LuxonisModel
<luxonis_train.core.core.LuxonisModel>` both read that file.

The config refers to each component by its registered name. The name
is usually the class name. A registry gets the class for the name, so
a new component needs no change to the training loop. See
`luxonis_train.registry` for the registries and
`luxonis_train.config.config` for the config schema.

``__version__`` holds the package version as a string, and
``__semver__`` holds it as a ``SemanticVersion``.

An import of the package also imports its submodules, so that the
built-in components register. After the submodule imports, the import
does these steps:

- It calls `luxonis_train.utils.setup_logging`.
- It adds `pathlib.Path`, `pathlib.PosixPath`, and
  `pathlib.WindowsPath` to the safe globals of
  ``torch.serialization``.

When the import of ``torch`` or of a submodule raises ``ImportError``,
the package issues a ``UserWarning``. It then skips the remaining
imports and these steps.

The package skips the submodule imports and these steps when
``sys.argv[0]`` ends with ``/luxonis_train``, so that the
``luxonis_train`` command starts fast. It does not skip them in these
cases:

- ``--source`` is on the command line.
- A reload of the package follows the skipped import. `create_model
  <luxonis_train.__main__.create_model>` does this reload for the
  commands that build a model.

"""

import sys
from typing import Final

from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.5.0"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)


# The first import from the CLI skips the submodule imports, so that
# the CLI starts fast.
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
