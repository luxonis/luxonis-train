"""PyDoctor extensions used to render Luxonis Train API documentation.

This module lives outside the ``luxonis_train`` package on purpose.
PyDoctor imports the class named by ``--system-class`` before it builds
the module tree, and importing a module of the documented package that
early makes PyDoctor fail when it later tries to document that same
module.
"""

from pydoctor.model import System
from pydoctor.options import Options

from tools.shape_contract import register_shape_contract_directive


class LuxonisTrainSystem(System):
    """PyDoctor system with Luxonis Train directives enabled."""

    def __init__(self, options: Options | None = None):
        register_shape_contract_directive()
        super().__init__(options)
