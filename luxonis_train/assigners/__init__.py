"""Assigner modules for matching predictions to ground truth targets."""

from .atss_assigner import ATSSAssigner
from .tal_assigner import TaskAlignedAssigner

__all__ = ["ATSSAssigner", "TaskAlignedAssigner"]
