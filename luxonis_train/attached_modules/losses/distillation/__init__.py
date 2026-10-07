"""Knowledge-distillation losses.

A distillation loss compares a node of the student with the matched
node of a teacher model. It sits in the ``distillation`` list of a
node, or the automatic recipe adds it when ``model.teacher`` is set.

- class logits: `LogitDistillationLoss`
- feature maps: `ChannelWiseDistillationLoss`

`BaseDistillationLoss` describes how to write a new one.

"""

from .base_distillation_loss import BaseDistillationLoss, StudentContext
from .channel_wise_distillation_loss import ChannelWiseDistillationLoss
from .logit_distillation_loss import LogitDistillationLoss

__all__ = [
    "BaseDistillationLoss",
    "ChannelWiseDistillationLoss",
    "LogitDistillationLoss",
    "StudentContext",
]
