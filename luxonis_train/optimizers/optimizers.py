import torch.optim as optim

from luxonis_train.utils.registry import OPTIMIZERS

from .custom_optimizers import TripleLRSGD

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
    TripleLRSGD,
]:
    OPTIMIZERS.register_module(module=optimizer)
