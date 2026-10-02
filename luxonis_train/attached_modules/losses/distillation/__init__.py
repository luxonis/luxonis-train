"""Knowledge-distillation losses.

A distillation loss compares a node of the student with the matched
node of a teacher model. It sits in the ``distillation`` list of a
node, or the automatic recipe adds it when ``model.teacher`` is set.

- class logits: `LogitKDLoss`
- feature maps: `CWDDistillationLoss`

`BaseDistillationLoss` describes how to write a new one.

"""

from .base_distillation_loss import BaseDistillationLoss
from .cwd_loss import CWDDistillationLoss
from .logit_kd_loss import LogitKDLoss

__all__ = ["BaseDistillationLoss", "CWDDistillationLoss", "LogitKDLoss"]
