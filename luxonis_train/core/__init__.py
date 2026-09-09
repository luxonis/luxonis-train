"""The entry point that drives training, testing, and export.

`LuxonisModel` builds the Lightning module and the loaders from a
config, and exposes ``train``, ``test``, ``export``, ``archive``,
``annotate``, ``infer``, and ``tune``. The ``luxonis_train`` command is
a thin wrapper over it.

"""

from .core import LuxonisModel

__all__ = ["LuxonisModel"]
