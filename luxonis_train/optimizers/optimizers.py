"""The registration of the ``torch.optim`` optimizers.

Importing the module registers these optimizers in the ``OPTIMIZERS``
registry, under their class names: ``Adadelta``, ``Adagrad``, ``Adam``,
``AdamW``, ``SparseAdam``, ``Adamax``, ``ASGD``, ``LBFGS``, ``NAdam``,
``RAdam``, ``RMSprop``, and ``SGD``.

"""

from torch import optim

from luxonis_train.registry import OPTIMIZERS

for optimizer in [
    optim.Adadelta,
    optim.Adagrad,
    optim.Adam,
    optim.AdamW,
    optim.SparseAdam,
    optim.Adamax,
    optim.ASGD,
    optim.LBFGS,
    optim.NAdam,
    optim.RAdam,
    optim.RMSprop,
    optim.SGD,
]:
    OPTIMIZERS.register(module=optimizer)
