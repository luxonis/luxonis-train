"""The entry point that drives training, testing, and export.

`LuxonisModel` builds the loaders, the trainer, and the Lightning module
from a config. Its methods ``train``, ``tune``, ``test``, ``infer``,
``annotate``, ``export``, ``archive``, ``convert``, and ``quantize`` run
the steps of a project. The commands of the ``luxonis_train`` CLI with
the same names build a `LuxonisModel` and call these methods.

"""

from .core import LuxonisModel

__all__ = ["LuxonisModel"]
