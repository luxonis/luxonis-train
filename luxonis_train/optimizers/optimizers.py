import torch.optim as optim

from luxonis_train.utils.registry import OPTIMIZERS

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
