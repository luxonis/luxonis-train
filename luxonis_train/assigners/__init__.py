from contextlib import suppress

from .atss_assigner import ATSSAssigner
from .tal_assigner import TaskAlignedAssigner

with suppress(ImportError):
    from aimet_torch.v2.nn import QuantizationMixin

    QuantizationMixin.ignore(ATSSAssigner)
    QuantizationMixin.ignore(TaskAlignedAssigner)


__all__ = ["ATSSAssigner", "TaskAlignedAssigner"]
