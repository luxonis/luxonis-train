"""The interface for a block that folds its training-time branches into
a single operation before export.
"""

from abc import ABC, abstractmethod

from torch import nn


class Reparameterizable(nn.Module, ABC):
    """Base class for a module that changes its structure for export.

    A subclass trains in one form, for example with parallel branches.
    `reparameterize` changes it to a form with fewer operations and the
    same output in the eval state. `restore` changes it back.
    `BaseNode.set_export_mode` calls `reparameterize` on every such
    submodule of the node for ``mode=True``, and `restore` for
    ``mode=False``. It does not check the current export mode.
    `LuxonisLightningModule.reparameterize` calls `reparameterize`
    without a change of the export mode. `GeneralReparameterizableBlock`
    is the subclass in this package.

    """

    @abstractmethod
    def reparameterize(self) -> None:
        """Change the module to its export form.

        An implementation changes the module in place. In the eval
        state, ``forward`` must then give the same output with fewer
        operations. `BaseNode.set_export_mode` calls the method on each
        call with ``mode=True``, also when export mode is already on. A
        call on a module in the export form must thus change nothing.

        """
        ...

    @abstractmethod
    def restore(self) -> None:
        """Change the module back to its training form.

        An implementation undoes `reparameterize`, so that ``forward``
        uses the training form again. `BaseNode.set_export_mode` calls
        the method on each call with ``mode=False``, also when export
        mode is already off. A call on a module in the training form
        must thus change nothing.

        """
        ...
