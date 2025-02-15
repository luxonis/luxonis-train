from abc import ABC, abstractmethod


class Reparametrizable(ABC):
    """An abstract class for reparametrizable modules.

    Reparametrizable modules are modules that support reparametrization
    of their parameters during export.

    Reparametrization is usually done to increase the performance of the
    model during inference by removing unnecessary parameters, fusing
    operations, and other methods.
    """

    @abstractmethod
    def reparametrize(self) -> None:
        """Reparametrize the module.

        This method is called before exporting the model. It is expected
        to be destructive and to modify the module's parameters in-
        place.
        """
        ...
