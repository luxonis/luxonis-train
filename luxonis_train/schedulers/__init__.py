"""Learning rate schedulers a config can name.

`luxonis_train.schedulers.schedulers` registers fifteen schedulers of
``torch.optim.lr_scheduler`` in the ``SCHEDULERS`` registry, under their
class names. ``trainer.scheduler.name`` accepts any of them.

Each inner optimizer of a training plan gets its own member scheduler.
When the plan has more than one inner optimizer, `CompositeLRScheduler`
steps the members of the `CompositeOptimizer` together.
`CompositeReduceLROnPlateau` does the same for the ``ReduceLROnPlateau``
members that monitor the same value.

"""

from .schedulers import *
