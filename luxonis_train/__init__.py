__version__ = "0.2.1"

import warnings

try:
    from .attached_modules import *
    from .core import *
    from .loaders import *
    from .models import *
    from .nodes import *
    from .optimizers import *
    from .schedulers import *
    from .strategies import *
    from .utils import *
except ImportError as e:
    warnings.warn(
        "Failed to import submodules. "
        "Some functionality of `luxonis-train` may be unavailable. "
        f"Error: `{e}`",
        stacklevel=2,
    )
