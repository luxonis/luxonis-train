from abc import ABC, abstractmethod

from torch import nn


class Reparametrizable(nn.Module, ABC):
    """An abstract class for reparametrizable modules.

    Reparametrizable modules are modules that support reparametrization
    of their parameters during export.

    Reparametrization is usually done to increase the performance of the
    model during inference by removing unnecessary parameters, fusing
    operations, and other methods.
    """

    @abstractmethod
    def reparametrize(self) -> None:
        """Reparametrizes the module.

        This method is typically called before exporting the model.
        """
        ...

    @abstractmethod
    def restore(self) -> None:
        """Restores the module to its original state."""
        ...
