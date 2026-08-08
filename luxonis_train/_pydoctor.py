"""PyDoctor extensions used to render Luxonis Train API documentation.

This module is named by ``system-class`` in the ``[tool.pydoctor]``
section of ``pyproject.toml``, so every PyDoctor invocation registers the
``shape-contract`` directive, including a bare ``uv run pydoctor``.

It is imported by PyDoctor and by nothing else, so the ``pydoctor`` and
``docutils`` dependencies it pulls in stay confined to the docs and dev
dependency groups. Do not import it from ``luxonis_train.__init__``.
"""

__docformat__ = "google"

from pydoctor.model import System
from pydoctor.options import Options

from luxonis_train._shape_contract import register_shape_contract_directive


class LuxonisTrainSystem(System):
    """PyDoctor system with Luxonis Train directives enabled."""

    def __init__(self, options: Options | None = None):
        register_shape_contract_directive()
        super().__init__(options)
