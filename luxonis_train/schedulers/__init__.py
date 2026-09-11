"""Learning rate schedulers a config can name.

Every scheduler of ``torch.optim.lr_scheduler`` is registered under its
class name, so ``trainer.scheduler.name`` accepts any of them.

`CompositeLRScheduler` steps the schedulers of a `CompositeOptimizer`
together, and `CompositeReduceLROnPlateau` does the same for the
schedulers that need a metric.

"""

from .schedulers import *
