__version__ = "0.3.0"

import shutil
import sys
import warnings

if not sys.argv or sys.argv[0] != shutil.which("luxonis_train"):
    try:
        from .attached_modules import *
        from .config import *
        from .core import *
        from .loaders import *
        from .models import *
        from .nodes import *
        from .optimizers import *
        from .schedulers import *
        from .strategies import *
        from .tasks import *
        from .utils import *
        from .utils import setup_logging

        setup_logging()

    except ImportError as e:
        warnings.warn(
            "Failed to import submodules. "
            "Some functionality of `luxonis-train` may be unavailable. "
            f"Error: `{e}`",
            stacklevel=2,
        )
