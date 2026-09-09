"""Optimizers a config can name.

Every optimizer of ``torch.optim`` is registered under its class name,
so ``trainer.optimizer.name`` accepts any of them and
``trainer.optimizer.params`` reaches its constructor.

`CompositeOptimizer` wraps the inner optimizers that node finetuning
rules and training strategies produce. It presents them to Lightning as
one optimizer, which keeps gradient accumulation and gradient clipping
working however many groups a config creates.

"""

from .optimizers import *
