"""The error a node raises when it cannot accept its input."""

__all__ = ["IncompatibleError"]


class IncompatibleError(Exception):
    """Raised when two parts of the model are incompatible with each
    other.
    """
