"""The error raised when two parts of the model cannot work together."""

__all__ = ["IncompatibleError"]


class IncompatibleError(Exception):
    """Error raised when two parts of the model cannot work together.

    A node raises it when the value of a property does not match the
    type annotation of that property in the node class. The error
    message names the predecessor of the node as a possible cause.

    An attached module raises it in two cases:

    - The task of its node is not in the supported tasks of the module.
    - Its node is not an instance of the ``node`` type annotation of
      the module.

    """
