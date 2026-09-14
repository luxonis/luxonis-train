"""Optimizers a config can name.

`luxonis_train.optimizers.optimizers` registers twelve optimizers of
``torch.optim`` in the ``OPTIMIZERS`` registry, under their class names.
``trainer.optimizer.name`` accepts any of them, and
``trainer.optimizer.params`` goes to its constructor.

`CompositeOptimizer` wraps the inner optimizers of a training plan when
the plan has more than one. Lightning then sees one optimizer, so
gradient accumulation and gradient clipping still work.

"""

from .optimizers import *
