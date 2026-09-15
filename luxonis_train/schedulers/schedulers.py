"""The registration of the ``torch.optim.lr_scheduler`` schedulers.

Importing the module registers these schedulers in the ``SCHEDULERS``
registry, under their class names: ``LambdaLR``, ``MultiplicativeLR``,
``StepLR``, ``MultiStepLR``, ``ConstantLR``, ``LinearLR``,
``ExponentialLR``, ``PolynomialLR``, ``CosineAnnealingLR``,
``ChainedScheduler``, ``SequentialLR``, ``ReduceLROnPlateau``,
``CyclicLR``, ``OneCycleLR``, and ``CosineAnnealingWarmRestarts``.

"""

from torch.optim import lr_scheduler

from luxonis_train.registry import SCHEDULERS

for scheduler in [
    lr_scheduler.LambdaLR,
    lr_scheduler.MultiplicativeLR,
    lr_scheduler.StepLR,
    lr_scheduler.MultiStepLR,
    lr_scheduler.ConstantLR,
    lr_scheduler.LinearLR,
    lr_scheduler.ExponentialLR,
    lr_scheduler.PolynomialLR,
    lr_scheduler.CosineAnnealingLR,
    lr_scheduler.ChainedScheduler,
    lr_scheduler.SequentialLR,
    lr_scheduler.ReduceLROnPlateau,
    lr_scheduler.CyclicLR,
    lr_scheduler.OneCycleLR,
    lr_scheduler.CosineAnnealingWarmRestarts,
]:
    SCHEDULERS.register(module=scheduler)
