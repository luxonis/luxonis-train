from abc import ABC, abstractmethod

from torch import nn


class Reparameterizable(nn.Module, ABC):
    """An abstract class for reparameterizable modules.

    Reparameterizable modules are modules that support
    reparameterization of their parameters during export.

    Reparameterization is usually done to increase the performance of
    the model during inference by removing unnecessary parameters,
    fusing operations, and other methods.
    """

    @abstractmethod
    def reparameterize(self) -> None:
        """Reparameterizes the module.

        This method is typically called before exporting the model.
        """
        ...

    @abstractmethod
    def restore(self) -> None:
        """Restores the module to its original state."""
        ...
